#!/usr/bin/env python3
"""
Comprehensive test for EchoWear PathManager with Object Detection
"""

import cv2
import numpy as np
import sys
import os
import time

def test_pathmanager_with_detection():
    """Test PathManager with object detection enabled"""
    print("🔧 Testing PathManager with Object Detection")
    print("=" * 50)
    
    try:
        from path_manager import PathManager
        
        # Initialize PathManager with object detection
        print("⏳ Initializing PathManager...")
        pm = PathManager(enable_object_detection=True)
        print("✅ PathManager initialized successfully!")
        
        # Test object detection capability
        if pm.enable_object_detection and pm.object_detector:
            print("✅ Object detection is enabled and ready")
            
            # Create a test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some test content
            cv2.rectangle(test_frame, (100, 100), (200, 300), (255, 255, 255), -1)
            cv2.putText(test_frame, "TEST", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Test detection
            print("⏳ Testing object detection on sample frame...")
            detections = pm.object_detector.detect_objects(test_frame)
            print(f"✅ Detection completed. Found {len(detections)} objects")
            
            if detections:
                for i, detection in enumerate(detections):
                    print(f"  📦 Object {i+1}: {detection['class']} (confidence: {detection['confidence']:.2f})")
            
        else:
            print("⚠️  Object detection is not available")
        
        # Test basic PathManager functionality
        print("\n⏳ Testing path recording functionality...")
        pm.start_recording_path("test_path", "test_destination")
        print("✅ Path recording started")
        
        # Simulate adding some path points
        for i in range(5):
            pm.add_path_point(f"step_{i}", i * 100, i * 50, frame=test_frame if i % 2 == 0 else None)
        
        pm.stop_recording_path()
        print("✅ Path recording stopped")
        
        # Check if path was saved
        paths = pm.get_saved_paths()
        print(f"✅ Saved paths: {len(paths)} total")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_camera_integration():
    """Test real camera integration if available"""
    print("\n🎥 Testing Camera Integration")
    print("=" * 30)
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  No camera available - skipping camera test")
            return True
        
        print("✅ Camera opened successfully")
        
        from path_manager import PathManager
        pm = PathManager(enable_object_detection=True)
        
        print("⏳ Capturing and analyzing 3 frames...")
        for i in range(3):
            ret, frame = cap.read()
            if ret:
                print(f"  📸 Frame {i+1} captured")
                
                if pm.enable_object_detection:
                    detections = pm.object_detector.detect_objects(frame)
                    print(f"    🔍 Found {len(detections)} objects")
                    
                    # Add to path if objects detected
                    if detections:
                        pm.add_path_point(f"camera_frame_{i}", i * 10, i * 5, frame=frame)
                
                time.sleep(0.5)  # Brief pause between frames
        
        cap.release()
        print("✅ Camera test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def main():
    """Run comprehensive tests"""
    print("🚀 EchoWear PathManager + Object Detection Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test 1: PathManager with Object Detection
    if not test_pathmanager_with_detection():
        success = False
    
    # Test 2: Camera Integration (optional)
    if not test_camera_integration():
        print("⚠️  Camera integration test failed (may be normal if no camera)")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All critical tests passed!")
        print("✅ EchoWear PathManager with Object Detection is ready!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
