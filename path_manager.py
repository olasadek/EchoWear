import cv2
import numpy as np
import os
import pickle
import json
import csv
from typing import List, Dict, Any, Optional, Tuple
import sys
import importlib.util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Import object detector
try:
    from camera_processor.object_detector import ObjectDetector, DetectionBackend
    OBJECT_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Object detection not available. Error: {e}")
    OBJECT_DETECTION_AVAILABLE = False

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def keypoints_to_serializable(keypoints):
    """Convert OpenCV KeyPoint objects to serializable format."""
    if keypoints is None:
        return None
    serializable_kp = []
    for kp in keypoints:
        serializable_kp.append({
            'pt': kp.pt,
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        })
    return serializable_kp

def keypoints_from_serializable(serializable_kp):
    """Convert serializable format back to OpenCV KeyPoint objects."""
    if serializable_kp is None:
        return None
    keypoints = []
    for kp_data in serializable_kp:
        kp = cv2.KeyPoint(
            x=kp_data['pt'][0],
            y=kp_data['pt'][1],
            size=kp_data['size'],
            angle=kp_data['angle'],
            response=kp_data['response'],
            octave=kp_data['octave'],
            class_id=kp_data['class_id']
        )
        keypoints.append(kp)
    return keypoints

def detections_to_serializable(detections):
    """Convert detection objects to serializable format."""
    if detections is None:
        return None
    serializable_detections = []
    for det in detections:
        serializable_detections.append({
            'label': det.label,
            'confidence': det.confidence,
            'bbox': det.bbox,
            'class_id': det.class_id,
            'timestamp': det.timestamp
        })
    return serializable_detections

class PathManager:
    def __init__(self, storage_dir="paths", enable_object_detection=True, 
                 nfeatures=1000, scaleFactor=1.2, nlevels=8,
                 min_matches=4,  # Reduced from 8
                 ransac_prob=0.95,  # Reduced from 0.999
                 ransac_threshold=2.0,  # Increased from 1.0
                 position_threshold=1.0,  # Increased from 0.5
                 orientation_threshold_large=0.5,  # Increased from 0.3 (~29 degrees)
                 orientation_threshold_small=0.2,  # Increased from 0.12 (~11 degrees)
                 loop_closure_threshold=0.3,  # Increased from 0.1
                 detection_confidence=0.3):  # Reduced from 0.5
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.recording = False
        self.current_path = []
        self.current_path_name = None
        self.current_destination = None
        self.loaded_path = None
        self.orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_kp = None
        self.last_desc = None
        self.last_pose = np.eye(4)  # 4x4 identity (start pose)
        self.poses = []
        self.current_pose = np.eye(4)  # Current estimated pose
        
        # Configurable thresholds for more flexible operation
        self.min_matches = min_matches
        self.ransac_prob = ransac_prob
        self.ransac_threshold = ransac_threshold
        self.position_threshold = position_threshold
        self.orientation_threshold_large = orientation_threshold_large
        self.orientation_threshold_small = orientation_threshold_small
        
        # Loop closure detection
        self.loop_closure_threshold = loop_closure_threshold
        self.pose_history = []
        
        # Object detection
        self.enable_object_detection = enable_object_detection and OBJECT_DETECTION_AVAILABLE
        self.last_detections = []
        if self.enable_object_detection:
            try:
                self.object_detector = ObjectDetector(
                    backend=DetectionBackend.OPENCV_DNN,
                    confidence_threshold=detection_confidence,
                    model_path="models"
                )
                print("Object detection enabled")
            except Exception as e:
                print(f"Failed to initialize object detector: {e}")
                self.enable_object_detection = False
        else:
            self.object_detector = None

    def start_recording_path(self, path_name: str, destination: str = None, coordinates=None):
        """Begin recording a new path with optional destination label and coordinates."""
        self.recording = True
        self.current_path = []
        self.current_path_name = path_name
        self.current_destination = destination
        self.current_coordinates = coordinates
        self.last_pose = np.eye(4)
        self.current_pose = np.eye(4)
        self.poses = []
        self.pose_history = []
        
        if destination:
            print(f"Started recording path: {path_name} → {destination}")
        else:
            print(f"Started recording path: {path_name}")

    def stop_recording_path(self):
        """Stop recording and save the path with destination label and coordinates."""
        if not self.recording or not self.current_path_name:
            print("No path is being recorded.")
            return
        
        # Convert keypoints to serializable format before saving
        serializable_path = []
        for frame_data in self.current_path:
            serializable_frame = {
                'keypoints': keypoints_to_serializable(frame_data['keypoints']),
                'descriptors': frame_data['descriptors'],
                'detections': detections_to_serializable(frame_data.get('detections', None))
            }
            serializable_path.append(serializable_frame)
        
        metadata = {
            "created_at": time.time(),
            "total_frames": len(self.current_path),
            "path_name": self.current_path_name
        }
        if hasattr(self, 'current_coordinates') and self.current_coordinates is not None:
            metadata["coordinates"] = self.current_coordinates
        
        path_data = {
            "frames": serializable_path,
            "poses": self.poses,
            "destination": self.current_destination,
            "metadata": metadata
        }
        path_file = os.path.join(self.storage_dir, f"{self.current_path_name}.pkl")
        with open(path_file, "wb") as f:
            pickle.dump(path_data, f)
        
        if self.current_destination:
            print(f"Saved path: {self.current_path_name} → {self.current_destination} ({len(self.current_path)} frames)")
        else:
            print(f"Saved path: {self.current_path_name} ({len(self.current_path)} frames)")
        
        self.recording = False
        self.current_path_name = None
        self.current_destination = None
        self.current_path = []
        self.poses = []
        self.current_coordinates = None

    def load_path(self, path_name: str):
        """Load a previously recorded path with destination information."""
        path_file = os.path.join(self.storage_dir, f"{path_name}.pkl")
        if not os.path.exists(path_file):
            print(f"Path {path_name} not found.")
            return False
        with open(path_file, "rb") as f:
            data = pickle.load(f)
        
        # Convert serializable keypoints back to OpenCV KeyPoint objects
        loaded_frames = []
        for frame_data in data['frames']:
            loaded_frame = {
                'keypoints': keypoints_from_serializable(frame_data['keypoints']),
                'descriptors': frame_data['descriptors'],
                'detections': frame_data.get('detections', None)  # Detections are already serializable
            }
            loaded_frames.append(loaded_frame)
        
        self.loaded_path = {
            'frames': loaded_frames,
            'poses': data['poses'],
            'destination': data.get('destination', None),
            'metadata': data.get('metadata', {})
        }
        
        destination_info = f" → {self.loaded_path['destination']}" if self.loaded_path['destination'] else ""
        print(f"Loaded path: {path_name}{destination_info} ({len(loaded_frames)} frames)")
        return True

    def start_navigation(self, path_name: str):
        """Initiate navigation along a loaded path."""
        if self.load_path(path_name):
            self.last_pose = np.eye(4)
            self.current_pose = np.eye(4)
            print(f"Navigation started for path: {path_name}")
            return True
        return False

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a frame for recording or localization. Returns current estimated pose."""
        kp, desc = self.orb.detectAndCompute(frame, None)
        
        # Object detection if enabled
        detections = None
        if self.enable_object_detection and self.object_detector:
            try:
                detection_result = self.object_detector.detect(frame)
                detections = detection_result.detections
                self.last_detections = detections
            except Exception as e:
                print(f"Object detection failed: {e}")
        
        if self.recording:
            frame_data = {
                "keypoints": kp, 
                "descriptors": desc,
                "detections": detections
            }
            self.current_path.append(frame_data)
            
            # For VO, estimate pose relative to last frame
            if self.last_kp is not None and self.last_desc is not None and desc is not None:
                matches = self.bf.match(self.last_desc, desc)
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) > self.min_matches:  # Use configurable threshold
                    src_pts = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.), 
                                                 method=cv2.RANSAC, prob=self.ransac_prob, 
                                                 threshold=self.ransac_threshold)
                    if E is not None:
                        _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts)
                        # Compose pose
                        pose = np.eye(4)
                        pose[:3, :3] = R
                        pose[:3, 3] = t.flatten()
                        self.last_pose = self.last_pose @ pose
                        self.current_pose = self.last_pose.copy()
                        self.poses.append(self.last_pose.copy())
                        self.pose_history.append(self.current_pose.copy())
                        
                        # Check for loop closure
                        self._check_loop_closure()
            
            self.last_kp, self.last_desc = kp, desc
        else:
            # Update current pose for navigation
            if self.last_kp is not None and self.last_desc is not None and desc is not None:
                matches = self.bf.match(self.last_desc, desc)
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) > self.min_matches:  # Use configurable threshold
                    src_pts = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.), 
                                                 method=cv2.RANSAC, prob=self.ransac_prob, 
                                                 threshold=self.ransac_threshold)
                    if E is not None:
                        _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts)
                        pose = np.eye(4)
                        pose[:3, :3] = R
                        pose[:3, 3] = t.flatten()
                        self.current_pose = self.current_pose @ pose
            
            self.last_kp, self.last_desc = kp, desc
        
        return self.current_pose

    def visualize_keypoints_and_matches(self, current_frame: np.ndarray, last_frame: np.ndarray = None) -> np.ndarray:
        """Visualize keypoints and matches between current and last frame."""
        if last_frame is None:
            # Just show keypoints for current frame
            kp, _ = self.orb.detectAndCompute(current_frame, None)
            result = current_frame.copy()
            cv2.drawKeypoints(current_frame, kp, result, color=(0, 255, 0))
            cv2.putText(result, f"Keypoints: {len(kp)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return result
        else:
            # Show matches between frames
            kp1, desc1 = self.orb.detectAndCompute(last_frame, None)
            kp2, desc2 = self.orb.detectAndCompute(current_frame, None)
            
            if desc1 is not None and desc2 is not None:
                matches = self.bf.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Draw matches
                result = cv2.drawMatches(last_frame, kp1, current_frame, kp2, matches[:20], None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.putText(result, f"Matches: {len(matches)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                return result
            else:
                return current_frame

    def get_guidance(self, current_frame: np.ndarray, current_pose: Optional[np.ndarray] = None) -> str:
        """
        Advanced guidance using feature matching to find closest frame in path.
        Now includes destination information in guidance and obstacle warnings.
        """
        obstacle_warning = ""
        if self.last_detections:
            obstacle_warning = "Warning, obstacle ahead. "
            try:
                # Provide direction based on the first detected object's position
                frame_width = current_frame.shape[1]
                frame_center_x = frame_width / 2
                
                detection = self.last_detections[0]
                
                # Ensure bbox is a tuple/list with at least 3 elements (x, y, w)
                if hasattr(detection, 'bbox') and len(detection.bbox) >= 3:
                    bbox_center_x = detection.bbox[0] + detection.bbox[2] / 2
                    
                    # Add a buffer zone in the center
                    if bbox_center_x < frame_center_x - (frame_width * 0.1):
                        obstacle_warning += f"{detection.label.capitalize()} on the left. "
                    elif bbox_center_x > frame_center_x + (frame_width * 0.1):
                        obstacle_warning += f"{detection.label.capitalize()} on the right. "
                    else:
                        obstacle_warning += f"{detection.label.capitalize()} directly ahead. "
            except (IndexError, AttributeError, TypeError):
                # Fallback to a generic warning if detection data is not as expected
                pass

        if self.loaded_path is None:
            return obstacle_warning + "No path loaded."
        
        if current_pose is None:
            current_pose = self.current_pose
        
        # Find closest frame in path using feature matching
        closest_frame_idx = self._find_closest_frame(current_frame)
        if closest_frame_idx is None:
            return obstacle_warning + "Unable to localize in path."
        
        # Get pose from closest frame
        if closest_frame_idx < len(self.loaded_path['poses']):
            path_pose = self.loaded_path['poses'][closest_frame_idx]
            
            # Calculate deviation
            position_deviation = np.linalg.norm(current_pose[:3, 3] - path_pose[:3, 3])
            
            # Calculate orientation difference
            current_rot = current_pose[:3, :3]
            path_rot = path_pose[:3, :3]
            orientation_diff = np.arccos(np.clip((np.trace(current_rot.T @ path_rot) - 1) / 2, -1, 1))
            
            # Generate guidance
            guidance = []
            if position_deviation > self.position_threshold:  # Use configurable threshold
                guidance.append("Position off")
            
            # Use only natural language for orientation
            if orientation_diff > self.orientation_threshold_small:  # Use configurable threshold
                # Use the new _get_directional_guidance
                direction = self._get_directional_guidance(current_pose, path_pose)
                guidance.append(direction)
            
            if not guidance:
                guidance.append("On track")
            
            # Add destination information if available
            destination = self.loaded_path.get('destination', None)
            if destination:
                guidance.append(f"Destination: {destination}")
            
            return obstacle_warning + ". ".join(guidance)
        else:
            destination = self.loaded_path.get('destination', None)
            if destination:
                return obstacle_warning + f"Path completed. You have reached: {destination}"
            else:
                return obstacle_warning + "Path completed."

    def _find_closest_frame(self, current_frame: np.ndarray) -> Optional[int]:
        """Find the closest frame in the loaded path using feature matching."""
        if self.loaded_path is None:
            return None
        
        current_kp, current_desc = self.orb.detectAndCompute(current_frame, None)
        if current_desc is None:
            return None
        
        best_match_idx = None
        best_match_score = 0
        
        for i, frame_data in enumerate(self.loaded_path['frames']):
            if frame_data['descriptors'] is not None:
                matches = self.bf.match(current_desc, frame_data['descriptors'])
                if len(matches) > 0:
                    # Use average match distance as score (lower is better)
                    avg_distance = np.mean([m.distance for m in matches])
                    score = len(matches) / (avg_distance + 1)  # Higher score is better
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = i
        
        return best_match_idx

    def _get_directional_guidance(self, current_pose: np.ndarray, target_pose: np.ndarray) -> str:
        """Get directional guidance based on pose differences."""
        # Get forward vectors
        current_forward = current_pose[:3, 0]  # X-axis is forward
        target_forward = target_pose[:3, 0]
        
        # Calculate angle between forward vectors
        angle = np.arccos(np.clip(np.dot(current_forward, target_forward), -1, 1))
        
        # Determine turn direction
        cross_product = np.cross(current_forward, target_forward)
        if cross_product[2] > 0:  # Positive Z means turn left
            direction = "left"
        else:
            direction = "right"
        
        # Lower threshold for turn, use natural language
        if angle > self.orientation_threshold_large:  # Use configurable threshold
            return f"turn {direction}"
        elif angle > self.orientation_threshold_small:  # Use configurable threshold
            return f"turn a bit {direction}"
        else:
            return "move forward"

    def _check_loop_closure(self):
        """Basic loop closure detection."""
        if len(self.pose_history) < 10:
            return
        
        current_pos = self.current_pose[:3, 3]
        
        # Check against recent poses
        for i, pose in enumerate(self.pose_history[-20:-5]):  # Skip very recent poses
            old_pos = pose[:3, 3]
            distance = np.linalg.norm(current_pos - old_pos)
            
            if distance < self.loop_closure_threshold:
                print(f"Loop closure detected! Distance: {distance:.3f}m")
                return

    def clear_all_paths(self):
        """Clear all saved paths from the storage directory."""
        if not os.path.exists(self.storage_dir):
            print("No paths directory found.")
            return
        
        files = [f for f in os.listdir(self.storage_dir) if f.endswith('.pkl')]
        if not files:
            print("No path files found to delete.")
            return
        
        for file in files:
            file_path = os.path.join(self.storage_dir, file)
            os.remove(file_path)
            print(f"Deleted: {file}")
        
        print(f"Cleared {len(files)} path files.")

    def export_path_to_json(self, path_name: str, output_file: str = None):
        """Export path data to JSON format."""
        if not self.load_path(path_name):
            return False
        
        if output_file is None:
            output_file = f"{path_name}_export.json"
        
        # Convert numpy arrays to lists and ensure all values are JSON serializable
        poses_list = []
        for pose in self.loaded_path['poses']:
            pose_list = []
            for row in pose:
                pose_list.append([float(x) for x in row])
            poses_list.append(pose_list)
        
        export_data = {
            "path_name": path_name,
            "total_frames": len(self.loaded_path['frames']),
            "poses": poses_list,
            "object_detections": []
        }
        
        for i, frame_data in enumerate(self.loaded_path['frames']):
            detections = frame_data.get('detections', None)
            if detections:
                frame_detections = []
                for det in detections:
                    frame_detections.append({
                        "label": det['label'],
                        "confidence": float(det['confidence']),
                        "bbox": [int(x) for x in det['bbox']],
                        "class_id": int(det['class_id'])
                    })
                export_data["object_detections"].append({
                    "frame": i,
                    "detections": frame_detections
                })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"Exported path to: {output_file}")
        return True

    def export_path_to_csv(self, path_name: str, output_file: str = None):
        """Export path data to CSV format."""
        if not self.load_path(path_name):
            return False
        
        if output_file is None:
            output_file = f"{path_name}_export.csv"
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'Pose_X', 'Pose_Y', 'Pose_Z', 'Objects'])
            
            for i, frame_data in enumerate(self.loaded_path['frames']):
                # Get pose data safely
                if i < len(self.loaded_path['poses']):
                    pose = self.loaded_path['poses'][i]
                    # Convert numpy array to list if needed
                    if hasattr(pose, 'tolist'):
                        pose = pose.tolist()
                    pose_x, pose_y, pose_z = pose[0][3], pose[1][3], pose[2][3]
                else:
                    pose_x, pose_y, pose_z = 0, 0, 0
                
                detections = frame_data.get('detections', None)
                
                objects_str = ""
                if detections:
                    object_labels = [det['label'] for det in detections]
                    objects_str = ", ".join(object_labels)
                
                writer.writerow([i, pose_x, pose_y, pose_z, objects_str])
        
        print(f"Exported path to: {output_file}")
        return True

    def visualize_path_and_current_pose(self, path_name: str = None):
        """Visualize the recorded path and current pose in 3D."""
        if path_name and not self.load_path(path_name):
            return
        
        if self.loaded_path is None:
            print("No path loaded for visualization.")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot path poses
        poses = self.loaded_path['poses']
        if poses:
            x_coords = [pose[0, 3] for pose in poses]
            y_coords = [pose[1, 3] for pose in poses]
            z_coords = [pose[2, 3] for pose in poses]
            
            ax.plot(x_coords, y_coords, z_coords, 'b-', label='Recorded Path', linewidth=2)
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], c='g', s=100, label='Start')
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], c='r', s=100, label='End')
        
        # Plot current pose
        current_pos = self.current_pose[:3, 3]
        ax.scatter(current_pos[0], current_pos[1], current_pos[2], c='orange', s=150, label='Current Position')
        
        # Draw coordinate frame for current pose
        scale = 0.1
        origin = current_pos
        x_axis = origin + scale * self.current_pose[:3, 0]
        y_axis = origin + scale * self.current_pose[:3, 1]
        z_axis = origin + scale * self.current_pose[:3, 2]
        
        ax.quiver(origin[0], origin[1], origin[2], 
                 x_axis[0] - origin[0], x_axis[1] - origin[1], x_axis[2] - origin[2], 
                 color='red', arrow_length_ratio=0.2, label='X')
        ax.quiver(origin[0], origin[1], origin[2], 
                 y_axis[0] - origin[0], y_axis[1] - origin[1], y_axis[2] - origin[2], 
                 color='green', arrow_length_ratio=0.2, label='Y')
        ax.quiver(origin[0], origin[1], origin[2], 
                 z_axis[0] - origin[0], z_axis[1] - origin[1], z_axis[2] - origin[2], 
                 color='blue', arrow_length_ratio=0.2, label='Z')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Path Visualization: {path_name or "Current Path"}')
        ax.legend()
        
        plt.show()

    def get_path_summary(self, path_name: str) -> Dict[str, Any]:
        """Get a summary of objects detected in a path."""
        if not self.load_path(path_name):
            return {"error": "Path not found"}
        
        object_counts = {}
        total_frames = len(self.loaded_path['frames'])
        frames_with_detections = 0
        
        for frame_data in self.loaded_path['frames']:
            detections = frame_data.get('detections', None)
            if detections:
                frames_with_detections += 1
                for det in detections:
                    label = det['label']
                    if label in object_counts:
                        object_counts[label] += 1
                    else:
                        object_counts[label] = 1
        
        return {
            "path_name": path_name,
            "total_frames": total_frames,
            "frames_with_detections": frames_with_detections,
            "object_counts": object_counts,
            "detection_rate": frames_with_detections / total_frames if total_frames > 0 else 0
        }

    def get_path_destination(self, path_name: str) -> Optional[str]:
        """Get the destination label for a specific path."""
        if not self.load_path(path_name):
            return None
        return self.loaded_path.get('destination', None)

    def list_paths_with_destinations(self) -> List[Dict[str, Any]]:
        """List all available paths with their destination labels."""
        if not os.path.exists(self.storage_dir):
            return []
        
        paths_info = []
        for file in os.listdir(self.storage_dir):
            if file.endswith('.pkl'):
                path_name = file.replace('.pkl', '')
                destination = self.get_path_destination(path_name)
                paths_info.append({
                    'name': path_name,
                    'destination': destination,
                    'file': file
                })
        return paths_info

def record_path_from_video(video_path: str, path_name: str, storage_dir: "paths"):
    """Example: Record a path from a video file."""
    path_manager = PathManager(storage_dir=storage_dir, enable_object_detection=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False
    
    path_manager.start_recording_path(path_name)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        path_manager.process_frame(frame)
        frame_count += 1
        
        # Show progress
        if frame_count % 30 == 0:  # Every 30 frames
            print(f"Processed {frame_count} frames...")
    
    path_manager.stop_recording_path()
    cap.release()
    print(f"Recorded path from video: {frame_count} frames")
    return True 