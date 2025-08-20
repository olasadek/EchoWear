import cv2
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional
import os

class DetectionBackend(Enum):
    """Enumeration for different detection backends"""
    YOLO = "yolo"
    MOBILENET = "mobilenet"
    OPENCV_DNN = "opencv_dnn"

class ObjectDetector:
    """
    Object detection class using YOLO or other CV models
    """
    
    def __init__(self, backend: DetectionBackend = DetectionBackend.YOLO, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the object detector
        
        Args:
            backend: Detection backend to use
            model_path: Path to model files directory
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.backend = backend
        self.model_path = model_path or "models"
        self.net = None
        self.classes = []
        self.colors = []
        self.conf_threshold = confidence_threshold
        self.nms_threshold = 0.4
        
        self._load_model()
    
    def _load_model(self):
        """Load the detection model"""
        if self.backend == DetectionBackend.YOLO:
            self._load_yolo_model()
        elif self.backend == DetectionBackend.OPENCV_DNN:
            self._load_yolo_model()  # Use YOLO as the DNN backend
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")
    
    def _load_yolo_model(self):
        """Load YOLO model"""
        try:
            # Construct paths to model files
            weights_path = os.path.join(self.model_path, "yolov4-tiny.weights")
            config_path = os.path.join(self.model_path, "yolov4-tiny.cfg")
            names_path = os.path.join(self.model_path, "coco.names")
            
            # Check if files exist
            if not all(os.path.exists(path) for path in [weights_path, config_path, names_path]):
                raise FileNotFoundError("YOLO model files not found")
            
            # Load the network
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Load class names
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            print(f"YOLO model loaded successfully with {len(self.classes)} classes")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Fallback to basic CV methods
            self._setup_fallback_detector()
    
    def _setup_fallback_detector(self):
        """Setup fallback detector using OpenCV's built-in methods"""
        try:
            # Use HOG person detector as fallback
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.classes = ["person"]  # Only person detection for fallback
            self.colors = [(0, 255, 0)]  # Green for person
            print("Using HOG person detector as fallback")
        except Exception as e:
            print(f"Error setting up fallback detector: {e}")
            self.net = None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in the given frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of detected objects with their properties
        """
        if self.net is not None:
            return self._detect_with_yolo(frame)
        elif hasattr(self, 'hog'):
            return self._detect_with_hog(frame)
        else:
            return []
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO"""
        height, width, channels = frame.shape
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        outs = self.net.forward(self._get_output_layers())
        
        # Parse detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                
                detections.append({
                    'class': label,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'center': [x + w // 2, y + h // 2],
                    'area': w * h
                })
        
        return detections
    
    def _detect_with_hog(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons using HOG descriptor (fallback)"""
        detections = []
        
        try:
            # Detect people
            boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))
            
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight > 0.5:  # Confidence threshold
                    detections.append({
                        'class': 'person',
                        'confidence': float(weight),
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'center': [int(x + w // 2), int(y + h // 2)],
                        'area': int(w * h)
                    })
        except Exception as e:
            print(f"Error in HOG detection: {e}")
        
        return detections
    
    def _get_output_layers(self):
        """Get output layers for YOLO"""
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = detection['class']
            confidence = detection['confidence']
            
            # Get color for this class
            if self.classes and label in self.classes:
                color_idx = self.classes.index(label)
                color = self.colors[color_idx]
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated_frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detections
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_objects': len(detections),
            'classes_detected': {},
            'average_confidence': 0.0
        }
        
        if detections:
            # Count objects by class
            for detection in detections:
                class_name = detection['class']
                if class_name in summary['classes_detected']:
                    summary['classes_detected'][class_name] += 1
                else:
                    summary['classes_detected'][class_name] = 1
            
            # Calculate average confidence
            total_confidence = sum(d['confidence'] for d in detections)
            summary['average_confidence'] = total_confidence / len(detections)
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Test the object detector
    detector = ObjectDetector(DetectionBackend.YOLO)
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 300), (255, 255, 255), -1)
    
    # Detect objects
    detections = detector.detect_objects(test_image)
    print(f"Detected {len(detections)} objects")
    
    # Draw detections
    annotated = detector.draw_detections(test_image, detections)
    
    # Get summary
    summary = detector.get_detection_summary(detections)
    print("Detection Summary:", summary)
