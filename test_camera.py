#!/usr/bin/env python3
"""
Simple Camera Test Script
Helps troubleshoot camera issues on Windows
"""

import cv2
import time
import sys

def test_camera():
    print("🔍 Testing camera access...")
    print("=" * 50)
    
    # Test different camera indices
    for camera_index in [0, 1, 2]:
        print(f"\n📷 Testing camera index {camera_index}...")
        
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"   ❌ Camera {camera_index}: Not available")
                continue
            
            print(f"   ✅ Camera {camera_index}: Opened successfully")
            
            # Try to read a few frames
            success_count = 0
            error_count = 0
            
            for i in range(10):
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success_count += 1
                        print(f"   Frame {i+1}: ✅ Success ({frame.shape})")
                    else:
                        error_count += 1
                        print(f"   Frame {i+1}: ❌ Failed to grab frame")
                except Exception as e:
                    error_count += 1
                    print(f"   Frame {i+1}: ❌ Error: {e}")
                
                time.sleep(0.1)
            
            print(f"   📊 Results: {success_count} successful, {error_count} failed")
            
            if success_count > 0:
                print(f"   🎉 Camera {camera_index} is working!")
                cap.release()
                return camera_index
            else:
                print(f"   ⚠️  Camera {camera_index} opened but can't grab frames")
            
            cap.release()
            
        except Exception as e:
            print(f"   ❌ Camera {camera_index}: Error - {e}")
    
    print("\n❌ No working camera found!")
    return None

def main():
    print("EchoWear Camera Test")
    print("=" * 50)
    
    # Check if OpenCV is available
    try:
        print(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV not available: {e}")
        return
    
    # Test camera
    working_camera = test_camera()
    
    if working_camera is not None:
        print(f"\n✅ Camera {working_camera} is working!")
        print("You can use this camera index in the GUI.")
    else:
        print("\n❌ No working camera found.")
        print("\nTroubleshooting tips:")
        print("1. Close other applications that might be using the camera")
        print("2. Check Windows camera permissions")
        print("3. Try restarting your computer")
        print("4. Check if your camera is enabled in Device Manager")

if __name__ == "__main__":
    main() 