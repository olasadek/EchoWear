#!/usr/bin/env python3
"""
Test Script for Path Manager

Tests the path recording, loading, and navigation functionality.
"""

import sys
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from path_manager import PathManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_frame(width=640, height=480, frame_num=0):
    """Create a test frame with some visual features."""
    # Create a base frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some geometric shapes that will create features
    # Rectangle
    cv2.rectangle(frame, (100 + frame_num*10, 100), (200 + frame_num*10, 200), (255, 0, 0), 2)
    
    # Circle
    cv2.circle(frame, (400, 300), 50, (0, 255, 0), 2)
    
    # Text
    cv2.putText(frame, f"Frame {frame_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def test_path_recording():
    """Test path recording functionality."""
    print("Testing Path Recording...")
    print("="*50)
    
    # Initialize path manager
    path_manager = PathManager(storage_dir="test_paths")
    
    # Start recording a path
    path_name = "test_path_1"
    path_manager.start_recording_path(path_name)
    
    # Simulate recording frames (moving camera)
    print("Recording frames...")
    for i in range(10):
        # Create a test frame with slight movement
        frame = create_test_frame(frame_num=i)
        
        # Process the frame
        path_manager.process_frame(frame)
        
        print(f"  Recorded frame {i+1}")
        time.sleep(0.1)  # Simulate real-time processing
    
    # Stop recording
    path_manager.stop_recording_path()
    
    print(f"✓ Path recording completed: {path_name}")
    return path_name

def test_path_loading():
    """Test path loading functionality."""
    print("\nTesting Path Loading...")
    print("="*50)
    
    path_manager = PathManager(storage_dir="test_paths")
    
    # Try to load the recorded path
    path_name = "test_path_1"
    success = path_manager.load_path(path_name)
    
    if success:
        print(f"✓ Successfully loaded path: {path_name}")
        print(f"  Frames: {len(path_manager.loaded_path['frames'])}")
        print(f"  Poses: {len(path_manager.loaded_path['poses'])}")
    else:
        print(f"✗ Failed to load path: {path_name}")
    
    return success

def test_navigation():
    """Test navigation functionality."""
    print("\nTesting Navigation...")
    print("="*50)
    
    path_manager = PathManager(storage_dir="test_paths")
    
    # Start navigation
    path_name = "test_path_1"
    success = path_manager.start_navigation(path_name)
    
    if not success:
        print(f"✗ Failed to start navigation for: {path_name}")
        return
    
    print(f"✓ Navigation started for: {path_name}")
    
    # Simulate navigation (current position)
    print("Testing guidance...")
    for i in range(5):
        # Create a test frame
        frame = create_test_frame(frame_num=i)
        
        # Process frame for localization
        path_manager.process_frame(frame)
        
        # Get guidance (simulate current pose)
        current_pose = np.eye(4)
        current_pose[0, 3] = i * 0.1  # Simulate moving forward
        current_pose[1, 3] = 0.2      # Simulate slight deviation
        
        guidance = path_manager.get_guidance(frame, current_pose)
        print(f"  Frame {i+1}: {guidance}")
    
    print("✓ Navigation test completed")

def test_path_listing():
    """Test listing available paths."""
    print("\nTesting Path Listing...")
    print("="*50)
    
    import os
    
    path_manager = PathManager(storage_dir="test_paths")
    
    # List all available paths
    if os.path.exists("test_paths"):
        paths = [f.replace('.pkl', '') for f in os.listdir("test_paths") if f.endswith('.pkl')]
        print(f"Available paths: {paths}")
        
        for path_name in paths:
            print(f"  - {path_name}")
    else:
        print("No paths directory found")

def test_error_handling():
    """Test error handling."""
    print("\nTesting Error Handling...")
    print("="*50)
    
    path_manager = PathManager(storage_dir="test_paths")
    
    # Test stopping recording when not recording
    print("Testing stop recording when not recording...")
    path_manager.stop_recording_path()
    
    # Test loading non-existent path
    print("Testing load non-existent path...")
    success = path_manager.load_path("non_existent_path")
    if not success:
        print("✓ Correctly handled non-existent path")
    
    # Test guidance without loaded path
    print("Testing guidance without loaded path...")
    frame = create_test_frame()
    guidance = path_manager.get_guidance(frame)
    print(f"  Guidance: {guidance}")

def run_all_tests():
    """Run all path manager tests."""
    print("Running Path Manager Tests...")
    print("="*60)
    
    try:
        # Test 1: Path Recording
        path_name = test_path_recording()
        
        # Test 2: Path Loading
        test_path_loading()
        
        # Test 3: Navigation
        test_navigation()
        
        # Test 4: Path Listing
        test_path_listing()
        
        # Test 5: Error Handling
        test_error_handling()
        
        print("\n" + "="*60)
        print("✓ All Path Manager tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        logger.error(f"Test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 