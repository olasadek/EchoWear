#!/usr/bin/env python3
"""
Test script for the Object Detector functionality
"""

import cv2
import numpy as np
import sys
import os

# Import the local camera processor
try:
    from camera_processor.object_detector import ObjectDetector, DetectionBackend
    print("✓ ObjectDetector imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ObjectDetector: {e}")
    sys.exit(1)

def test_object_detector():
    """Test the object detector functionality"""
    print("\n=== Testing Object Detector ===")
    
    # Initialize detector
    try:
        detector = ObjectDetector(DetectionBackend.YOLO, model_path="models")
        print("✓ ObjectDetector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize ObjectDetector: {e}")
        return False
    
    # Create a test image (simple colored rectangle)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored shapes to simulate objects
    cv2.rectangle(test_image, (100, 100), (200, 300), (255, 255, 255), -1)  # White rectangle
    cv2.circle(test_image, (400, 200), 50, (0, 255, 0), -1)  # Green circle
    
    print("✓ Test image created")
    
    # Test object detection
    try:
        detections = detector.detect_objects(test_image)
        print(f"✓ Object detection completed. Found {len(detections)} objects")
        
        # Print detection details
        for i, detection in enumerate(detections):
            print(f"  Object {i+1}: {detection['class']} (confidence: {detection['confidence']:.2f})")
        
    except Exception as e:
        print(f"✗ Object detection failed: {e}")
        return False
    
    # Test drawing detections
    try:
        annotated = detector.draw_detections(test_image, detections)
        print("✓ Detection drawing completed")
    except Exception as e:
        print(f"✗ Detection drawing failed: {e}")
        return False
    
    # Test summary generation
    try:
        summary = detector.get_detection_summary(detections)
        print("✓ Detection summary generated")
        print(f"  Summary: {summary}")
    except Exception as e:
        print(f"✗ Summary generation failed: {e}")
        return False
    
    return True

def test_camera_capture():
    """Test camera capture (if available)"""
    print("\n=== Testing Camera Capture ===")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠ No camera available for testing")
            return True
        
        print("✓ Camera opened successfully")
        
        # Capture a frame
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured successfully (size: {frame.shape})")
            
            # Test detection on real frame
            detector = ObjectDetector(DetectionBackend.YOLO, model_path="models")
            detections = detector.detect_objects(frame)
            print(f"✓ Real-time detection completed. Found {len(detections)} objects")
        else:
            print("⚠ Failed to capture frame")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 EchoWear Object Detector Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    if not test_object_detector():
        print("\n❌ Basic object detector tests failed")
        return 1
    
    # Test camera (optional)
    test_camera_capture()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\nThe object detector is ready for use in EchoWear.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
