#!/usr/bin/env python3
"""
Enhanced PathManager Demo with Scene Graph Detection

Demonstrates all the new features added to PathManager with scene graph integration:
- Keypoint visualization with scene descriptions
- Advanced guidance with scene graph obstacle detection and TTS
- 3D path visualization with scene understanding
- Path export and analysis with scene data
- Path management with scene graph metadata
- Video recording with scene graph building
- GPS navigation with scene graph enhanced obstacle detection

Scene Graph Integration:
- Demo 1: Records paths with real-time scene descriptions and object detection
- Demo 2: Uses scene graphs for intelligent obstacle detection and voice guidance
- Demo 5: Creates multiple paths with scene graph data collection
- Demo 6: GPS navigation enhanced with scene graph obstacle detection

Scene Graph Components:
- HierarchicalGraphBuilder: Builds hierarchical scene representations
- SceneGraphBuilder: Creates scene graphs from object detections
- GraphStore: Stores and manages scene graph data
- Object detection integration with YOLO models
- Natural language scene descriptions
- Intelligent obstacle classification
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import pyttsx3
import threading
import math
import geocoder  # pip install geocoder

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from path_manager import PathManager
from geocoding_utils import get_coordinates_from_place, find_nearby_places

def calculate_distance_and_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance and initial bearing between two points.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (in degrees)
        lat2, lon2: Latitude and longitude of second point (in degrees)
    
    Returns:
        tuple: (distance_meters, bearing_degrees)
    """
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula for distance
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in meters
    R = 6371000
    distance_meters = R * c
    
    # Calculate bearing
    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    bearing_rad = math.atan2(y, x)
    bearing_degrees = math.degrees(bearing_rad)
    
    # Convert to 0-360 range
    bearing_degrees = (bearing_degrees + 360) % 360
    
    return distance_meters, bearing_degrees

def get_cardinal_direction(bearing_degrees):
    """
    Convert bearing degrees to cardinal direction.
    
    Args:
        bearing_degrees: Bearing in degrees (0-360)
    
    Returns:
        str: Cardinal direction (e.g., "North", "North-East", "East")
    """
    # Define cardinal directions with their bearing ranges
    directions = [
        (0, 22.5, "North"),
        (22.5, 67.5, "North-East"),
        (67.5, 112.5, "East"),
        (112.5, 157.5, "South-East"),
        (157.5, 202.5, "South"),
        (202.5, 247.5, "South-West"),
        (247.5, 292.5, "West"),
        (292.5, 337.5, "North-West"),
        (337.5, 360, "North")
    ]
    
    for min_bearing, max_bearing, direction in directions:
        if min_bearing <= bearing_degrees < max_bearing:
            return direction
    
    return "North"  # Default fallback

def demo_1_visualization():
    """Demo 1: Keypoint visualization during recording with destination labeling."""
    print("Demo 1: Keypoint Visualization with Destination Labeling")
    print("="*50)
    
    # Initialize scene graph detection
    print("🔍 Initializing scene graph detection...")
    scene_graph_available = False
    try:
        # Add the llm-camera-tracker directory to Python path
        camera_tracker_path = os.path.join(os.path.dirname(__file__), "llm-camera-tracker")
        if camera_tracker_path not in sys.path:
            sys.path.insert(0, camera_tracker_path)
        
        # Import scene graph components
        from scene_graph.hierarchical_graph_builder import HierarchicalGraphBuilder
        from scene_graph.graph_builder import SceneGraphBuilder
        from scene_graph.graph_store import GraphStore
        
        # Initialize scene graph components
        graph_builder = HierarchicalGraphBuilder()
        scene_builder = SceneGraphBuilder()
        graph_store = GraphStore()
        scene_graph_available = True
        print("✅ Scene graph detection initialized")
        
    except Exception as e:
        print(f"⚠️  Scene graph not available: {e}")
        print("   Using basic object detection instead")
        scene_graph_available = False
    
    path_manager = PathManager(storage_dir="demo_paths", enable_object_detection=True)
    
    # Ask user for destination
    destination = input("Enter destination for this path (e.g., 'kitchen', 'bedroom'): ").strip()
    if not destination:
        destination = None
    
    # Ask user for place name for geocoding (optional)
    place_name = input("Enter a place name for this path (optional, for geocoding): ").strip()
    coordinates = None
    if place_name:
        coordinates = get_coordinates_from_place(place_name)
        if coordinates:
            print(f"Coordinates for '{place_name}': {coordinates}")
        else:
            print(f"Could not geocode '{place_name}'. Proceeding without coordinates.")
    
    path_manager.start_recording_path("visualization_demo", destination, coordinates)
    
    print("Recording with keypoint visualization and scene graph detection. Press 'q' or close window to stop.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available. Using simulated frames.")
        for i in range(30):
            frame = create_test_frame(i)
            pose = path_manager.process_frame(frame)
            
            # Scene graph detection
            obstacle_message = ""
            scene_description = ""
            if scene_graph_available:
                try:
                    if path_manager.enable_object_detection and path_manager.object_detector:
                        detection_result = path_manager.object_detector.detect(frame)
                        if detection_result and detection_result.detections:
                            detected_objects = []
                            for detection in detection_result.detections:
                                if detection.confidence > 0.5:
                                    detected_objects.append(detection.label)
                            
                            if detected_objects:
                                if len(detected_objects) == 1:
                                    scene_description = f"There is a {detected_objects[0]} in the scene."
                                else:
                                    scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                                
                                try:
                                    scene_graph, action_graph, object_graph = graph_builder.update_scene_state(
                                        scene_description, 
                                        time.time()
                                    )
                                    
                                    best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                    if best_detection.confidence > 0.6:
                                        obstacle_message = f"{best_detection.label} detected"
                                        
                                except Exception as graph_error:
                                    print(f"Scene graph update error: {graph_error}")
                                    
                except Exception as e:
                    print(f"Scene graph detection error: {e}")
            
            vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            
            # Add scene graph info overlay
            if scene_graph_available:
                cv2.putText(vis_frame, "Scene Graph Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if scene_description:
                    cv2.putText(vis_frame, f"Scene: {scene_description}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if obstacle_message:
                    cv2.putText(vis_frame, f"Detection: {obstacle_message}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Keypoint Visualization with Scene Graph", vis_frame)
            
            # Improved key detection
            key = cv2.waitKey(100) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping recording")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Keypoint Visualization with Scene Graph", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed - stopping recording")
                    break
            except:
                pass
    else:
        last_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose = path_manager.process_frame(frame)
            
            # Scene graph detection
            obstacle_message = ""
            scene_description = ""
            if scene_graph_available:
                try:
                    if path_manager.enable_object_detection and path_manager.object_detector:
                        detection_result = path_manager.object_detector.detect(frame)
                        if detection_result and detection_result.detections:
                            detected_objects = []
                            for detection in detection_result.detections:
                                if detection.confidence > 0.5:
                                    detected_objects.append(detection.label)
                            
                            if detected_objects:
                                if len(detected_objects) == 1:
                                    scene_description = f"There is a {detected_objects[0]} in the scene."
                                else:
                                    scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                                
                                try:
                                    scene_graph, action_graph, object_graph = graph_builder.update_scene_state(
                                        scene_description, 
                                        time.time()
                                    )
                                    
                                    best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                    if best_detection.confidence > 0.6:
                                        obstacle_message = f"{best_detection.label} detected"
                                        
                                except Exception as graph_error:
                                    print(f"Scene graph update error: {graph_error}")
                                    
                except Exception as e:
                    print(f"Scene graph detection error: {e}")
            
            if last_frame is not None:
                vis_frame = path_manager.visualize_keypoints_and_matches(frame, last_frame)
            else:
                vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            
            # Add scene graph info overlay
            if scene_graph_available:
                cv2.putText(vis_frame, "Scene Graph Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if scene_description:
                    cv2.putText(vis_frame, f"Scene: {scene_description}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if obstacle_message:
                    cv2.putText(vis_frame, f"Detection: {obstacle_message}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Keypoint Visualization with Scene Graph", vis_frame)
            last_frame = frame.copy()
            
            # Improved key detection
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping recording")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Keypoint Visualization with Scene Graph", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed - stopping recording")
                    break
            except:
                pass
        
        cap.release()
    
    cv2.destroyAllWindows()
    path_manager.stop_recording_path()
    print("✓ Visualization demo completed")

def demo_2_advanced_guidance():
    """Demo 2: Advanced guidance with feature matching and TTS."""
    print("\nDemo 2: Advanced Guidance with TTS and Destination")
    print("="*50)
    
    # Initialize TTS engine with better settings
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.8)  # Volume level
        print("✅ TTS engine initialized successfully")
        tts_available = True
    except Exception as e:
        print(f"❌ Failed to initialize TTS engine: {e}")
        print("   Voice guidance will be disabled")
        engine = None
        tts_available = False
    
    # Global TTS lock to prevent run loop issues
    tts_lock = threading.Lock()
    
    def speak_safely(text):
        """Safely speak text without run loop conflicts."""
        if tts_available and engine:
            try:
                with tts_lock:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def speak_async(text):
        """Speak text asynchronously."""
        if tts_available and engine:
            t = threading.Thread(target=speak_safely, args=(text,))
            t.daemon = True
            t.start()
    
    # Initialize scene graph detection
    print("🔍 Initializing scene graph detection...")
    scene_graph_available = False
    try:
        # Add the llm-camera-tracker directory to Python path
        camera_tracker_path = os.path.join(os.path.dirname(__file__), "llm-camera-tracker")
        if camera_tracker_path not in sys.path:
            sys.path.insert(0, camera_tracker_path)
        
        # Import scene graph components
        from scene_graph.hierarchical_graph_builder import HierarchicalGraphBuilder
        from scene_graph.graph_builder import SceneGraphBuilder
        from scene_graph.graph_store import GraphStore
        
        # Initialize scene graph components
        graph_builder = HierarchicalGraphBuilder()
        scene_builder = SceneGraphBuilder()
        graph_store = GraphStore()
        scene_graph_available = True
        print("✅ Scene graph detection initialized")
        
    except Exception as e:
        print(f"⚠️  Scene graph not available: {e}")
        print("   Using basic object detection instead")
        scene_graph_available = False
    
    path_manager = PathManager(storage_dir="demo_paths")
    
    # Multi-path selection with geocoding
    paths_info = path_manager.list_paths_with_destinations()
    if not paths_info:
        print("No recorded paths found. Run Demo 1 or 5 first.")
        return
    
    # Prompt for place name
    place_name = input("Enter a place name to navigate to (or leave blank to list all paths): ").strip()
    selected_path = None
    if place_name:
        coords = get_coordinates_from_place(place_name)
        if coords:
            print(f"Coordinates for '{place_name}': {coords}")
            # Find closest path with coordinates in metadata
            min_dist = float('inf')
            closest_path = None
            for path_info in paths_info:
                # Try to load path metadata
                meta_coords = None
                try:
                    if path_manager.load_path(path_info['name']):
                        meta = path_manager.loaded_path.get('metadata', {})
                        meta_coords = meta.get('coordinates', None)
                except Exception:
                    pass
                if meta_coords:
                    # Compute Euclidean distance in lat/lon
                    dist = math.sqrt((coords[0] - meta_coords[0])**2 + (coords[1] - meta_coords[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_path = path_info['name']
            # Threshold for 'nearby' (about 1km)
            if closest_path and min_dist < 0.01:
                print(f"Navigating closest path: {closest_path} (distance: {min_dist:.4f} deg)")
                selected_path = closest_path
            else:
                print("No path found near that location. Showing all available paths.")
        else:
            print("Could not geocode that place. Showing all available paths.")
    
    if not selected_path:
        print("\nAvailable paths:")
        for i, path_info in enumerate(paths_info, 1):
            destination = path_info['destination'] or "No destination"
            print(f"  {i}. {path_info['name']} → {destination}")
        while True:
            try:
                selection = int(input(f"Select a path to navigate (1-{len(paths_info)}): "))
                if 1 <= selection <= len(paths_info):
                    break
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")
        selected_path = paths_info[selection-1]['name']
        print(f"\nNavigating path: {selected_path}")
    
    if not path_manager.load_path(selected_path):
        print("Failed to load selected path.")
        return
    path_manager.start_navigation(selected_path)
    
    print("Testing advanced guidance with TTS and scene graph detection. Press 'q' or close window to stop.")
    
    # Welcome message and destination announcement
    if tts_available:
        destination_name = selected_path.replace('_', ' ').replace('path to ', '')
        welcome_message = f"Starting navigation to {destination_name}. Voice guidance is active."
        print(f"🔊 {welcome_message}")
        speak_async(welcome_message)
        time.sleep(1)  # Give time for the message to be spoken
    
    destination_announced = False
    camera_covered = False
    camera_covered_announced = False
    localization_lost_announced = False
    last_obstacle_message = ""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available. Using simulated frames.")
        for i in range(20):
            frame = create_test_frame(i)
            pose = path_manager.process_frame(frame)

            # Camera covered detection
            mean_brightness = np.mean(frame)
            if mean_brightness < 15:
                if not camera_covered_announced:
                    speak_async("Camera is covered.")
                    print("Camera is covered.")
                    camera_covered_announced = True
                camera_covered = True
                guidance = "Camera is covered."
            else:
                if camera_covered:
                    camera_covered = False
                    camera_covered_announced = False
                
                # Scene graph obstacle detection
                obstacle_message = ""
                scene_description = ""
                if scene_graph_available:
                    try:
                        # Use object detection from path_manager to get detections
                        if path_manager.enable_object_detection and path_manager.object_detector:
                            detection_result = path_manager.object_detector.detect(frame)
                            if detection_result and detection_result.detections:
                                # Create a scene description from detections
                                detected_objects = []
                                for detection in detection_result.detections:
                                    if detection.confidence > 0.5:
                                        detected_objects.append(detection.label)
                                
                                if detected_objects:
                                    # Create a natural language description of the scene
                                    if len(detected_objects) == 1:
                                        scene_description = f"There is a {detected_objects[0]} in the scene."
                                    else:
                                        scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                                    
                                    # Update scene graph with the description
                                    try:
                                        scene_graph, action_graph, object_graph = graph_builder.update_scene_state(
                                            scene_description, 
                                            time.time()
                                        )
                                        
                                        # Get the most important detection for obstacle warning
                                        best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                        if best_detection.confidence > 0.6:
                                            obstacle_message = f"{best_detection.label} detected ahead"
                                            
                                            # Check if it's a person (highest priority)
                                            if best_detection.label.lower() in ['person', 'people', 'man', 'woman', 'child']:
                                                obstacle_message = f"Person detected ahead - be careful"
                                            
                                    except Exception as graph_error:
                                        print(f"Scene graph update error: {graph_error}")
                                        # Fallback to basic detection
                                        best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                        if best_detection.confidence > 0.5:
                                            obstacle_message = f"{best_detection.label} detected ahead"

                    except Exception as e:
                        print(f"Scene graph detection error: {e}")
                        # Fallback to basic object detection
                        if path_manager.enable_object_detection and path_manager.object_detector:
                            detection_result = path_manager.object_detector.detect(frame)
                            if detection_result and detection_result.detections:
                                best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                if best_detection.confidence > 0.5:
                                    obstacle_message = f"{best_detection.label} detected ahead"

                # Fallback to basic guidance if no scene graph or no detections
                if not obstacle_message:
                    guidance = path_manager.get_guidance(frame, pose)
                else:
                    # Only announce obstacle if it changed
                    if obstacle_message != last_obstacle_message:
                        speak_async(obstacle_message)
                        last_obstacle_message = obstacle_message
                    guidance = f"{obstacle_message}. {path_manager.get_guidance(frame, pose)}"

            # Lost localization detection
            if guidance == "Unable to localize in path.":
                if not localization_lost_announced:
                    speak_async("Localization lost.")
                    print("Localization lost.")
                    localization_lost_announced = True
                # Show message but don't repeat
                guidance = "Localization lost."
            else:
                localization_lost_announced = False

            # Only announce destination the first time
            guidance_to_speak = guidance
            if "Destination:" in guidance:
                if destination_announced:
                    guidance_to_speak = ". ".join([s for s in guidance.split('. ') if not s.startswith("Destination:")])
                else:
                    destination_announced = True
            
            # Add active TTS guidance for regular navigation
            if tts_available and guidance_to_speak and guidance_to_speak != "Camera is covered.":
                # Speak guidance instructions (but not too frequently)
                if not hasattr(demo_2_advanced_guidance, 'last_guidance_time'):
                    demo_2_advanced_guidance.last_guidance_time = 0
                    demo_2_advanced_guidance.last_guidance_text = ""
                
                current_time = time.time()
                # Only speak if guidance changed and enough time has passed (3 seconds)
                if (guidance_to_speak != demo_2_advanced_guidance.last_guidance_text and 
                    current_time - demo_2_advanced_guidance.last_guidance_time > 3.0):
                    speak_async(guidance_to_speak)
                    demo_2_advanced_guidance.last_guidance_time = current_time
                    demo_2_advanced_guidance.last_guidance_text = guidance_to_speak
            
            vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            
            # Add scene graph info overlay
            if scene_graph_available:
                cv2.putText(vis_frame, "Scene Graph Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if scene_description:
                    cv2.putText(vis_frame, f"Scene: {scene_description}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if obstacle_message:
                    cv2.putText(vis_frame, f"Obstacle: {obstacle_message}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    guidance_y = 120
                else:
                    guidance_y = 90
            else:
                guidance_y = 90
            
            # Add TTS status
            if tts_available:
                cv2.putText(vis_frame, "Voice Guidance Active", (10, guidance_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                guidance_y += 60
            else:
                cv2.putText(vis_frame, "Voice Guidance Disabled", (10, guidance_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                guidance_y += 60
            
            cv2.putText(vis_frame, guidance, (10, guidance_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Advanced Guidance with Scene Graph", vis_frame)
            
            print(f"Frame {i+1}: {guidance}")
            
            # Stop if destination is reached
            if "Path completed" in guidance:
                completion_message = f"Destination reached. Navigation complete."
                print(f"🎉 {completion_message}")
                if tts_available:
                    speak_async(completion_message)
                time.sleep(2) # Give time to see final frame and hear audio
                break

            # Improved key detection
            key = cv2.waitKey(500) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping guidance")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Advanced Guidance with Scene Graph", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed - stopping guidance")
                    break
            except:
                pass
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose = path_manager.process_frame(frame)

            # Camera covered detection
            mean_brightness = np.mean(frame)
            if mean_brightness < 15:
                if not camera_covered_announced:
                    speak_async("Camera is covered.")
                    print("Camera is covered.")
                    camera_covered_announced = True
                camera_covered = True
                guidance = "Camera is covered."
            else:
                if camera_covered:
                    camera_covered = False
                    camera_covered_announced = False
                
                # Scene graph obstacle detection
                obstacle_message = ""
                scene_description = ""
                if scene_graph_available:
                    try:
                        # Use object detection from path_manager to get detections
                        if path_manager.enable_object_detection and path_manager.object_detector:
                            detection_result = path_manager.object_detector.detect(frame)
                            if detection_result and detection_result.detections:
                                # Create a scene description from detections
                                detected_objects = []
                                for detection in detection_result.detections:
                                    if detection.confidence > 0.5:
                                        detected_objects.append(detection.label)
                                
                                if detected_objects:
                                    # Create a natural language description of the scene
                                    if len(detected_objects) == 1:
                                        scene_description = f"There is a {detected_objects[0]} in the scene."
                                    else:
                                        scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                                    
                                    # Update scene graph with the description
                                    try:
                                        scene_graph, action_graph, object_graph = graph_builder.update_scene_state(
                                            scene_description, 
                                            time.time()
                                        )
                                        
                                        # Get the most important detection for obstacle warning
                                        best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                        if best_detection.confidence > 0.6:
                                            obstacle_message = f"{best_detection.label} detected ahead"
                                            
                                            # Check if it's a person (highest priority)
                                            if best_detection.label.lower() in ['person', 'people', 'man', 'woman', 'child']:
                                                obstacle_message = f"Person detected ahead - be careful"
                                            
                                    except Exception as graph_error:
                                        print(f"Scene graph update error: {graph_error}")
                                        # Fallback to basic detection
                                        best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                        if best_detection.confidence > 0.5:
                                            obstacle_message = f"{best_detection.label} detected ahead"

                    except Exception as e:
                        print(f"Scene graph detection error: {e}")
                        # Fallback to basic object detection
                        if path_manager.enable_object_detection and path_manager.object_detector:
                            detection_result = path_manager.object_detector.detect(frame)
                            if detection_result and detection_result.detections:
                                best_detection = max(detection_result.detections, key=lambda x: x.confidence)
                                if best_detection.confidence > 0.5:
                                    obstacle_message = f"{best_detection.label} detected ahead"

                # Fallback to basic guidance if no scene graph or no detections
                if not obstacle_message:
                    guidance = path_manager.get_guidance(frame, pose)
                else:
                    # Only announce obstacle if it changed
                    if obstacle_message != last_obstacle_message:
                        speak_async(obstacle_message)
                        last_obstacle_message = obstacle_message
                    guidance = f"{obstacle_message}. {path_manager.get_guidance(frame, pose)}"

            # Lost localization detection
            if guidance == "Unable to localize in path.":
                if not localization_lost_announced:
                    speak_async("Localization lost.")
                    print("Localization lost.")
                    localization_lost_announced = True
                # Show message but don't repeat
                guidance = "Localization lost."
            else:
                localization_lost_announced = False

            # Only announce destination the first time
            guidance_to_speak = guidance
            if "Destination:" in guidance:
                if destination_announced:
                    guidance_to_speak = ". ".join([s for s in guidance.split('. ') if not s.startswith("Destination:")])
                else:
                    destination_announced = True
            
            # Add active TTS guidance for regular navigation
            if tts_available and guidance_to_speak and guidance_to_speak != "Camera is covered.":
                # Speak guidance instructions (but not too frequently)
                if not hasattr(demo_2_advanced_guidance, 'last_guidance_time'):
                    demo_2_advanced_guidance.last_guidance_time = 0
                    demo_2_advanced_guidance.last_guidance_text = ""
                
                current_time = time.time()
                # Only speak if guidance changed and enough time has passed (3 seconds)
                if (guidance_to_speak != demo_2_advanced_guidance.last_guidance_text and 
                    current_time - demo_2_advanced_guidance.last_guidance_time > 3.0):
                    speak_async(guidance_to_speak)
                    demo_2_advanced_guidance.last_guidance_time = current_time
                    demo_2_advanced_guidance.last_guidance_text = guidance_to_speak
            
            vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            
            # Add scene graph info overlay
            if scene_graph_available:
                cv2.putText(vis_frame, "Scene Graph Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if scene_description:
                    cv2.putText(vis_frame, f"Scene: {scene_description}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if obstacle_message:
                    cv2.putText(vis_frame, f"Obstacle: {obstacle_message}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    guidance_y = 120
                else:
                    guidance_y = 90
            else:
                guidance_y = 90
            
            # Add TTS status
            if tts_available:
                cv2.putText(vis_frame, "Voice Guidance Active", (10, guidance_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                guidance_y += 60
            else:
                cv2.putText(vis_frame, "Voice Guidance Disabled", (10, guidance_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                guidance_y += 60
            
            cv2.putText(vis_frame, guidance, (10, guidance_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Advanced Guidance with Scene Graph", vis_frame)
            
            # Stop if destination is reached
            if "Path completed" in guidance:
                completion_message = f"Destination reached. Navigation complete."
                print(f"🎉 {completion_message}")
                if tts_available:
                    speak_async(completion_message)
                time.sleep(2) # Give time to see final frame and hear audio
                break

            # Improved key detection
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping guidance")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Advanced Guidance with Scene Graph", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed - stopping guidance")
                    break
            except:
                pass
        
        cap.release()
    
    cv2.destroyAllWindows()
    print("✓ Advanced guidance demo completed")

def demo_3_export_and_analysis():
    """Demo 3: Path export and analysis."""
    print("\nDemo 3: Export and Analysis")
    print("="*40)
    
    path_manager = PathManager(storage_dir="demo_paths")
    
    # Get path summary
    summary = path_manager.get_path_summary("visualization_demo")
    
    if "error" not in summary:
        print(f"Path Summary:")
        print(f"  Total Frames: {summary['total_frames']}")
        print(f"  Frames with Detections: {summary['frames_with_detections']}")
        print(f"  Detection Rate: {summary['detection_rate']:.2%}")
        
        if summary['object_counts']:
            print("\nDetected Objects:")
            for obj, count in summary['object_counts'].items():
                print(f"  {obj}: {count} detections")
        
        # Export to JSON
        print("\nExporting to JSON...")
        path_manager.export_path_to_json("visualization_demo", "demo_export.json")
        
        # Export to CSV
        print("Exporting to CSV...")
        path_manager.export_path_to_csv("visualization_demo", "demo_export.csv")
        
        print("✓ Export completed")
    else:
        print(f"Error: {summary['error']}")

def demo_4_path_management():
    """Demo 4: Path management operations with destination labeling."""
    print("\nDemo 4: Path Management with Destinations")
    print("="*50)
    
    path_manager = PathManager(storage_dir="demo_paths")
    
    # List all paths with destinations
    print("Available paths with destinations:")
    paths_info = path_manager.list_paths_with_destinations()
    
    if paths_info:
        for i, path_info in enumerate(paths_info, 1):
            destination = path_info['destination'] or "No destination"
            print(f"  {i}. {path_info['name']} → {destination}")
    else:
        print("  No paths found")
    
    # Show destination for specific path
    if paths_info:
        print(f"\nDestination details:")
        for path_info in paths_info:
            destination = path_manager.get_path_destination(path_info['name'])
            if destination:
                print(f"  {path_info['name']} leads to: {destination}")
            else:
                print(f"  {path_info['name']}: No destination set")
    
    # Note: clear_all_paths is commented out for safety
    # print("\nTo clear all paths, uncomment the line in the code:")
    # print("path_manager.clear_all_paths()")

def demo_5_destination_labeling():
    """Demo 5: Create multiple paths with different destinations."""
    print("\nDemo 5: Multiple Paths with Destinations")
    print("="*50)
    
    # Initialize scene graph detection
    print("🔍 Initializing scene graph detection...")
    scene_graph_available = False
    try:
        # Add the llm-camera-tracker directory to Python path
        camera_tracker_path = os.path.join(os.path.dirname(__file__), "llm-camera-tracker")
        if camera_tracker_path not in sys.path:
            sys.path.insert(0, camera_tracker_path)
        
        # Import scene graph components
        from scene_graph.hierarchical_graph_builder import HierarchicalGraphBuilder
        from scene_graph.graph_builder import SceneGraphBuilder
        from scene_graph.graph_store import GraphStore
        
        # Initialize scene graph components
        graph_builder = HierarchicalGraphBuilder()
        scene_builder = SceneGraphBuilder()
        graph_store = GraphStore()
        scene_graph_available = True
        print("✅ Scene graph detection initialized")
        
    except Exception as e:
        print(f"⚠️  Scene graph not available: {e}")
        print("   Using basic object detection instead")
        scene_graph_available = False
    
    path_manager = PathManager(storage_dir="demo_paths", enable_object_detection=True)
    
    # Create a few example paths with destinations
    destinations = ["kitchen", "bedroom", "office"]
    
    for i, destination in enumerate(destinations):
        print(f"\nRecording path to {destination} with scene graph detection...")
        path_name = f"path_to_{destination}"
        
        path_manager.start_recording_path(path_name, destination)
        
        # Simulate recording (shorter for demo)
        for j in range(10):
            frame = create_test_frame(j + i * 10)
            pose = path_manager.process_frame(frame)
            
            # Scene graph detection
            if scene_graph_available:
                try:
                    if path_manager.enable_object_detection and path_manager.object_detector:
                        detection_result = path_manager.object_detector.detect(frame)
                        if detection_result and detection_result.detections:
                            detected_objects = []
                            for detection in detection_result.detections:
                                if detection.confidence > 0.5:
                                    detected_objects.append(detection.label)
                            
                            if detected_objects:
                                if len(detected_objects) == 1:
                                    scene_description = f"There is a {detected_objects[0]} in the scene."
                                else:
                                    scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                                
                                try:
                                    scene_graph, action_graph, object_graph = graph_builder.update_scene_state(
                                        scene_description, 
                                        time.time()
                                    )
                                    print(f"  Scene: {scene_description}")
                                        
                                except Exception as graph_error:
                                    print(f"Scene graph update error: {graph_error}")
                                    
                except Exception as e:
                    print(f"Scene graph detection error: {e}")
            
            time.sleep(0.1)
        
        path_manager.stop_recording_path()
    
    print("\n✓ Created paths with destinations and scene graph data:")
    paths_info = path_manager.list_paths_with_destinations()
    for path_info in paths_info:
        destination = path_info['destination'] or "No destination"
        print(f"  {path_info['name']} → {destination}")

def create_test_frame(frame_num=0):
    """Create a test frame with moving features."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add moving rectangle
    x_offset = 100 + frame_num * 10
    cv2.rectangle(frame, (x_offset, 100), (x_offset + 100, 200), (255, 0, 0), 2)
    
    # Add circle
    cv2.circle(frame, (400, 300), 50, (0, 255, 0), 2)
    
    # Add text
    cv2.putText(frame, f"Frame {frame_num}", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add noise
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def demo_6_gps_search():
    """Demo 6: GPS-based location search with camera integration and obstacle detection."""
    print("\nDemo 6: GPS Navigation with Camera & Obstacle Detection")
    print("="*60)
    
    print("Attempting to get laptop's current location...")
    
    # Check if geocoder is available
    try:
        import geocoder
    except ImportError:
        print("❌ geocoder package not installed. Installing it...")
        print("   Run: pip install geocoder")
        print("   Then restart the demo.")
        return
    
    # Initialize camera and object detection
    print("\n📷 Initializing camera for obstacle detection...")
    camera_available = True  # We'll check camera availability later when needed
    
    # Initialize scene graph detection
    print("🔍 Initializing scene graph detection...")
    scene_graph_available = False
    try:
        # Add the llm-camera-tracker directory to Python path
        camera_tracker_path = os.path.join(os.path.dirname(__file__), "llm-camera-tracker")
        if camera_tracker_path not in sys.path:
            sys.path.insert(0, camera_tracker_path)
        
        # Import scene graph components
        from scene_graph.hierarchical_graph_builder import HierarchicalGraphBuilder
        from scene_graph.graph_builder import SceneGraphBuilder
        from scene_graph.graph_store import GraphStore
        
        # Initialize scene graph components
        graph_builder = HierarchicalGraphBuilder()
        scene_builder = SceneGraphBuilder()
        graph_store = GraphStore()
        scene_graph_available = True
        print("✅ Scene graph detection initialized")
        
    except Exception as e:
        print(f"⚠️  Scene graph not available: {e}")
        print("   Using basic object detection instead")
        scene_graph_available = False
    
    # Initialize TTS
    print("🔊 Initializing voice guidance...")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        print("✅ Voice guidance initialized")
        tts_available = True
    except Exception as e:
        print(f"⚠️  Voice guidance not available: {e}")
        tts_available = False
        engine = None
    
    # Global TTS lock to prevent run loop issues
    tts_lock = threading.Lock()
    
    def speak_safely(text):
        """Safely speak text without run loop conflicts."""
        if tts_available and engine:
            try:
                with tts_lock:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")

    try:
        # Try multiple geocoding methods for better reliability
        latitude, longitude = None, None
        
        # Method 1: IP-based geocoding
        try:
            print("   Trying IP-based location detection...")
            g = geocoder.ip('me')
            if g.latlng is not None:
                latitude, longitude = g.latlng[0], g.latlng[1]
                print(f"✅ IP-based location detected: {latitude}, {longitude}")
        except Exception as e:
            print(f"   IP-based detection failed: {e}")
        
        # Method 2: Try WMI-based detection (Windows)
        if latitude is None:
            try:
                print("   Trying WMI-based location detection...")
                g = geocoder.wmi('me')
                if g.latlng is not None:
                    latitude, longitude = g.latlng[0], g.latlng[1]
                    print(f"✅ WMI-based location detected: {latitude}, {longitude}")
            except Exception as e:
                print(f"   WMI-based detection failed: {e}")
        
        # Method 3: Use a default location if all methods fail
        if latitude is None:
            print("❌ Could not automatically detect location.")
            print("   Using default location (New York City) for demonstration...")
            latitude, longitude = 40.7580, -73.9855  # Times Square, NYC
            print(f"✅ Using default location: {latitude}, {longitude}")
        
        # Get user query for place type
        print("\nEnter a type of place to search for:")
        print("Examples: 'cafe', 'hospital', 'park', 'restaurant', 'bank'")
        print("Or press Enter to search for general places")
        
        user_query = input("Your search query: ").strip()
        
        if not user_query:
            print("No query provided. Searching for general places...")
            user_query = None
        
        print(f"\nSearching for places within 1km radius...")
        print("This may take a few seconds...")
        
        # Perform search using geocoding_utils with timeout handling
        try:
            places = find_nearby_places(latitude, longitude, query=user_query, distance_km=1)
            
            if places:
                print(f"\n✅ Found {len(places)} places:")
                for i, place in enumerate(places, 1):
                    name = place.get('name', 'N/A')
                    address = place.get('address', 'N/A')
                    distance = place.get('distance_km', 'N/A')
                    print(f"  {i}. {name} ({distance} km)")
                    print(f"     Address: {address}")
                    print()
                
                # Navigation functionality
                print("Navigation Options:")
                print("  Enter a number to navigate to that place")
                print("  Enter '0' to cancel navigation")
                
                try:
                    selection = input("Enter the number of the place to navigate to (or '0' to cancel): ").strip()
                    
                    if selection == '0':
                        print("Navigation cancelled.")
                        return
                    
                    selection_num = int(selection)
                    if 1 <= selection_num <= len(places):
                        target_place = places[selection_num - 1]
                        target_name = target_place.get('name', 'Unknown location')
                        target_coords = target_place.get('coordinates', None)
                        
                        if target_coords:
                            target_lat, target_lon = target_coords
                            print(f"\n🎯 Navigating to: {target_name}")
                            print(f"📍 Target coordinates: {target_lat}, {target_lon}")
                            
                            # Navigation loop
                            print("\n🚶 Starting navigation with obstacle detection...")
                            print("Press Ctrl+C to stop navigation")
                            
                            last_guidance_message = ""
                            last_guidance_time = 0
                            
                            # Initialize camera display
                            cap = cv2.VideoCapture(0)
                            if not cap.isOpened():
                                print("No camera available. Navigation will continue without obstacle detection.")
                                cap = None
                            else:
                                print("📷 Camera initialized for obstacle detection")
                                cv2.namedWindow("Navigation with Obstacle Detection", cv2.WINDOW_NORMAL)
                                cv2.resizeWindow("Navigation with Obstacle Detection", 800, 600)
                            
                            try:
                                while True:
                                    current_time = time.time()
                                    
                                    # Get current location
                                    try:
                                        g = geocoder.ip('me')
                                        if g.latlng is not None:
                                            current_lat, current_lon = g.latlng[0], g.latlng[1]
                                            
                                            # Calculate distance and bearing
                                            distance, bearing = calculate_distance_and_bearing(
                                                current_lat, current_lon, target_lat, target_lon
                                            )
                                            
                                            # Check if arrived
                                            if distance < 20:  # 20 meters threshold
                                                arrival_message = f"You have arrived at your destination: {target_name}!"
                                                print(f"🎉 {arrival_message}")
                                                speak_safely(arrival_message)
                                                break
                                            
                                            # Get camera frame and detect obstacles
                                            obstacle_message = ""
                                            display_frame = None
                                            scene_description = ""
                                            
                                            if cap and cap.isOpened():
                                                ret, frame = cap.read()
                                                if ret:
                                                    display_frame = frame.copy()
                                                    
                                                    # Scene graph obstacle detection (priority)
                                                    if scene_graph_available:
                                                        try:
                                                            # Use object detection from path_manager to get detections
                                                            # For demo 6, we'll use basic OpenCV detection since we don't have path_manager
                                                            # But we can still use scene graph for analysis
                                                            
                                                            # Basic object detection using contours
                                                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                                                            edges = cv2.Canny(blurred, 50, 150)
                                                            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                                            
                                                            # Analyze center region
                                                            height, width = gray.shape
                                                            center_x, center_y = width // 2, height // 2
                                                            center_region_size = min(width, height) // 4
                                                            center_left = center_x - center_region_size
                                                            center_right = center_x + center_region_size
                                                            center_top = center_y - center_region_size
                                                            center_bottom = center_y + center_region_size
                                                            
                                                            # Find significant objects in center region
                                                            detected_objects = []
                                                            for contour in contours:
                                                                area = cv2.contourArea(contour)
                                                                x, y, w, h = cv2.boundingRect(contour)
                                                                
                                                                if (center_left < x < center_right and 
                                                                    center_top < y < center_bottom and
                                                                    area > 500):
                                                                    
                                                                    # Classify object based on shape and size
                                                                    aspect_ratio = h / w if w > 0 else 0
                                                                    if 0.3 < aspect_ratio < 4.0:  # Human-like proportions
                                                                        detected_objects.append("person")
                                                                    elif area > 2000:  # Large object
                                                                        detected_objects.append("large object")
                                                                    else:  # Small object
                                                                        detected_objects.append("small object")
                                                                    
                                                                    # Draw detection box
                                                                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                                    cv2.putText(display_frame, f"Object ({area:.0f})", (x, y - 10), 
                                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                                            
                                                            # Create scene description and update scene graph
                                                            if detected_objects:
                                                                if len(detected_objects) == 1:
                                                                    scene_description = f"There is a {detected_objects[0]} in the scene."
                                                                else:
                                                                    scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                                                                
                                                                try:
                                                                    scene_graph, action_graph, object_graph = graph_builder.update_scene_state(
                                                                        scene_description, 
                                                                        time.time()
                                                                    )
                                                                    
                                                                    # Get the most important detection for obstacle warning
                                                                    if "person" in detected_objects:
                                                                        obstacle_message = "Person detected ahead - be careful"
                                                                    elif "large object" in detected_objects:
                                                                        obstacle_message = "Large object detected ahead"
                                                                    elif "small object" in detected_objects:
                                                                        obstacle_message = "Small object detected ahead"
                                                                        
                                                                except Exception as graph_error:
                                                                    print(f"Scene graph update error: {graph_error}")
                                                                    # Fallback to basic detection
                                                                    if detected_objects:
                                                                        obstacle_message = f"{detected_objects[0]} detected ahead"
                                                                        
                                                        except Exception as e:
                                                            print(f"Scene graph detection error: {e}")
                                                    
                                                    # Enhanced obstacle detection (fallback/backup)
                                                    if not obstacle_message:
                                                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                                        
                                                        # 1. Motion detection (compare with previous frame)
                                                        if not hasattr(demo_6_gps_search, 'prev_frame'):
                                                            demo_6_gps_search.prev_frame = gray.copy()
                                                        
                                                        # Calculate frame difference
                                                        frame_diff = cv2.absdiff(gray, demo_6_gps_search.prev_frame)
                                                        motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                                                        
                                                        # 2. Contour detection for objects
                                                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                                                        edges = cv2.Canny(blurred, 50, 150)
                                                        
                                                        # Find contours
                                                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                                        
                                                        # 3. Analyze center region for obstacles
                                                        height, width = gray.shape
                                                        center_x, center_y = width // 2, height // 2
                                                        center_region_size = min(width, height) // 4
                                                        
                                                        # Define center region
                                                        center_left = center_x - center_region_size
                                                        center_right = center_x + center_region_size
                                                        center_top = center_y - center_region_size
                                                        center_bottom = center_y + center_region_size
                                                        
                                                        # Check for large contours in center region
                                                        obstacle_detected = False
                                                        for contour in contours:
                                                            # Get contour area and bounding box
                                                            area = cv2.contourArea(contour)
                                                            x, y, w, h = cv2.boundingRect(contour)
                                                            
                                                            # Check if contour is in center region and large enough
                                                            if (center_left < x < center_right and 
                                                                center_top < y < center_bottom and
                                                                area > 500):  # Lowered from 1000 to 500
                                                                
                                                                # Check aspect ratio (humans are typically taller than wide)
                                                                aspect_ratio = h / w if w > 0 else 0
                                                                if 0.3 < aspect_ratio < 4.0:  # Wider range for human-like proportions
                                                                    obstacle_detected = True
                                                                    obstacle_message = "Human or large object detected ahead"
                                                                    
                                                                    # Draw detection box
                                                                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                                                    cv2.putText(display_frame, f"Object ({area:.0f})", (x, y - 10), 
                                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                                                    break
                                                        
                                                        # 4. Check for motion in center region
                                                        if not obstacle_detected:
                                                            center_motion = motion_mask[center_top:center_bottom, center_left:center_right]
                                                            motion_pixels = np.sum(center_motion > 0)
                                                            motion_ratio = motion_pixels / center_motion.size
                                                            
                                                            if motion_ratio > 0.02:  # Lowered from 0.05 to 0.02 (2% motion threshold)
                                                                obstacle_detected = True
                                                                
                                                                # Try to identify the type of motion based on characteristics
                                                                if motion_ratio > 0.1:  # High motion - likely person
                                                                    obstacle_message = "Person detected ahead"
                                                                elif motion_ratio > 0.05:  # Medium motion - could be person or object
                                                                    obstacle_message = "Moving object detected ahead"
                                                                else:  # Low motion - small object
                                                                    obstacle_message = "Small motion detected ahead"
                                                                
                                                                # Draw motion region
                                                                cv2.rectangle(display_frame, (center_left, center_top), 
                                                                             (center_right, center_bottom), (0, 165, 255), 2)
                                                                cv2.putText(display_frame, f"Motion ({motion_ratio:.2f})", 
                                                                           (center_left, center_top - 10), 
                                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                                        
                                                        # 5. Check for dark regions (potential obstacles)
                                                        if not obstacle_detected:
                                                            center_region = gray[center_top:center_bottom, center_left:center_right]
                                                            mean_brightness = np.mean(center_region)
                                                            brightness_std = np.std(center_region)
                                                            
                                                            # Dark region with high contrast (like a person in dark clothes)
                                                            if mean_brightness < 100 and brightness_std > 20:  # Relaxed thresholds
                                                                obstacle_detected = True
                                                                obstacle_message = "Dark object detected ahead"
                                                                
                                                                cv2.rectangle(display_frame, (center_left, center_top), 
                                                                             (center_right, center_bottom), (128, 0, 128), 2)
                                                                cv2.putText(display_frame, f"Dark Object", 
                                                                           (center_left, center_top - 10), 
                                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
                                                        
                                                        # Update previous frame
                                                        demo_6_gps_search.prev_frame = gray.copy()
                                                    
                                                    # Add navigation info overlay
                                                    cv2.putText(display_frame, f"Distance: {int(distance)}m", (10, 30), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                                    cv2.putText(display_frame, f"Direction: {get_cardinal_direction(bearing)}", (10, 60), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                                    cv2.putText(display_frame, f"Target: {target_name}", (10, 90), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                                    
                                                    # Add scene graph info overlay
                                                    if scene_graph_available:
                                                        cv2.putText(display_frame, "Scene Graph Active", (10, 120), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                                        if scene_description:
                                                            cv2.putText(display_frame, f"Scene: {scene_description}", (10, 150), 
                                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                                        if obstacle_message:
                                                            cv2.putText(display_frame, f"Obstacle: {obstacle_message}", (10, 180), 
                                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                                            info_y = 210
                                                        else:
                                                            info_y = 180
                                                    else:
                                                        info_y = 120
                                                    
                                                    # Add detection info
                                                    if obstacle_message:
                                                        cv2.putText(display_frame, "OBSTACLE DETECTED!", (10, info_y), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                                    else:
                                                        # Show debug info when no obstacle detected
                                                        cv2.putText(display_frame, "No obstacles detected", (10, info_y), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                                        cv2.putText(display_frame, f"Contours: {len(contours)}", (10, info_y + 30), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                                        if 'motion_ratio' in locals():
                                                            cv2.putText(display_frame, f"Motion: {motion_ratio:.3f}", (10, info_y + 50), 
                                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                                        if 'mean_brightness' in locals():
                                                            cv2.putText(display_frame, f"Brightness: {mean_brightness:.0f}", (10, info_y + 70), 
                                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                                    
                                                    cv2.imshow("Navigation with Scene Graph & Obstacle Detection", display_frame)
                                                    
                                                    # Check for window close or key press
                                                    key = cv2.waitKey(1) & 0xFF
                                                    if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                                                        print("Key pressed - stopping navigation")
                                                        break
                                                    try:
                                                        if cv2.getWindowProperty("Navigation with Obstacle Detection", cv2.WND_PROP_VISIBLE) < 1:
                                                            print("Window closed - stopping navigation")
                                                            break
                                                    except:
                                                        pass
                                            
                                            # Provide guidance (only when obstacle detected or guidance changes)
                                            should_speak = False
                                            if obstacle_message:
                                                guidance_message = f"{obstacle_message}. Head {get_cardinal_direction(bearing)}, approximately {int(distance)} meters to {target_name}"
                                                should_speak = True
                                            else:
                                                guidance_message = f"Head {get_cardinal_direction(bearing)}, approximately {int(distance)} meters to {target_name}"
                                                # Only speak if this is a significant change in direction or distance
                                                if not hasattr(demo_6_gps_search, 'last_direction') or not hasattr(demo_6_gps_search, 'last_distance'):
                                                    demo_6_gps_search.last_direction = get_cardinal_direction(bearing)
                                                    demo_6_gps_search.last_distance = int(distance)
                                                    should_speak = True  # First time speaking
                                                else:
                                                    # Check if direction or distance changed significantly
                                                    current_direction = get_cardinal_direction(bearing)
                                                    current_distance = int(distance)
                                                    distance_diff = abs(current_distance - demo_6_gps_search.last_distance)
                                                    
                                                    if (current_direction != demo_6_gps_search.last_direction or 
                                                        distance_diff > 50):  # 50 meter change threshold
                                                        should_speak = True
                                                        demo_6_gps_search.last_direction = current_direction
                                                        demo_6_gps_search.last_distance = current_distance
                                            
                                            # Only speak if guidance changed or obstacle detected
                                            if should_speak and guidance_message != last_guidance_message:
                                                print(f"🧭 {guidance_message}")
                                                speak_safely(guidance_message)
                                                last_guidance_message = guidance_message
                                            
                                        else:
                                            lost_message = "Lost GPS signal, trying to re-acquire..."
                                            print(f"⚠️  {lost_message}")
                                            speak_safely(lost_message)
                                            time.sleep(5)
                                            
                                    except Exception as location_error:
                                        print(f"⚠️  Error getting location: {location_error}")
                                        time.sleep(5)
                                        
                            except KeyboardInterrupt:
                                print("\n⚠️  Navigation stopped by user.")
                                speak_safely("Navigation stopped")
                            finally:
                                # Stop camera and close windows
                                if cap:
                                    cap.release()
                                cv2.destroyAllWindows()
                        else:
                            print("❌ Target location coordinates not available.")
                    else:
                        print("❌ Invalid selection. Please enter a number between 1 and", len(places))
                        
                except ValueError:
                    print("❌ Invalid input. Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n⚠️  Navigation cancelled by user.")
            else:
                print("❌ No places found in the specified area.")
                print("   This might be due to:")
                print("   - Limited data in the area")
                print("   - Network connectivity issues")
                print("   - Geocoding service limitations")
                
        except Exception as search_error:
            print(f"❌ Error during place search: {search_error}")
            print("   This might be due to network issues or service unavailability.")
            print("   Try again later or check your internet connection.")
                
    except KeyboardInterrupt:
        print("\n⚠️  Search cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error during GPS search: {e}")
        print("   This might be due to:")
        print("   - Network connectivity issues")
        print("   - Geocoding service unavailability")
        print("   - System configuration problems")
        import traceback
        traceback.print_exc()

def run_all_demos():
    """Run all PathManager demos."""
    print("Enhanced PathManager Features Demo with Scene Graph Detection")
    print("="*70)
    
    try:
        # Create demo paths directory
        os.makedirs("demo_paths", exist_ok=True)
        
        # Run demos with scene graph detection
        demo_1_visualization()
        demo_2_advanced_guidance()
        demo_3_export_and_analysis()
        demo_4_path_management()
        demo_5_destination_labeling()
        
        # Run demo 6 with error handling to prevent crashes
        try:
            demo_6_gps_search()
        except Exception as e:
            print(f"\n⚠️  Demo 6 (GPS Search) failed: {e}")
            print("   This is likely due to network issues or missing geocoder package.")
            print("   Run 'pip install geocoder' and try again.")
            print("   Continuing with other demos...")
        
        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("\nGenerated files:")
        print("  - demo_export.json (path data)")
        print("  - demo_export.csv (path data)")
        print("  - demo_paths/ (recorded paths with destinations)")
        print("\nScene Graph Features:")
        print("  - Demo 1: Scene descriptions during path recording")
        print("  - Demo 2: Scene graph obstacle detection with TTS")
        print("  - Demo 5: Scene graph data collection for multiple paths")
        print("  - Demo 6: Scene graph enhanced GPS navigation")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_demos() 