#!/usr/bin/env python3
"""
Enhanced PathManager Demo

Demonstrates all the new features added to PathManager:
- Keypoint visualization
- Advanced guidance with feature matching
- 3D path visualization
- Path export and analysis
- Path management
- Video recording
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import pyttsx3
import threading
import math

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from path_manager import PathManager
from geocoding_utils import get_coordinates_from_place

def demo_1_visualization():
    """Demo 1: Keypoint visualization during recording with destination labeling."""
    print("Demo 1: Keypoint Visualization with Destination Labeling")
    print("="*50)
    
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
    
    print("Recording with keypoint visualization. Press 'q' or close window to stop.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available. Using simulated frames.")
        for i in range(30):
            frame = create_test_frame(i)
            pose = path_manager.process_frame(frame)
            vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            cv2.imshow("Keypoint Visualization", vis_frame)
            
            # Improved key detection
            key = cv2.waitKey(100) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping recording")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Keypoint Visualization", cv2.WND_PROP_VISIBLE) < 1:
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
            
            if last_frame is not None:
                vis_frame = path_manager.visualize_keypoints_and_matches(frame, last_frame)
            else:
                vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            
            cv2.imshow("Keypoint Visualization", vis_frame)
            last_frame = frame.copy()
            
            # Improved key detection
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping recording")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Keypoint Visualization", cv2.WND_PROP_VISIBLE) < 1:
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
    
    # Initialize TTS engine
    try:
        engine = pyttsx3.init()
        print("TTS engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize TTS engine: {e}")
        engine = None
    
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
    
    print("Testing advanced guidance with TTS. Press 'q' or close window to stop.")
    
    def speak_async(text):
        if engine:
            t = threading.Thread(target=lambda: engine.say(text) or engine.runAndWait())
            t.daemon = True
            t.start()

    destination_announced = False
    camera_covered = False
    camera_covered_announced = False
    localization_lost_announced = False
    
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
                guidance = path_manager.get_guidance(frame, pose)

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
            
            # Speak the guidance asynchronously
            if not camera_covered_announced:
                speak_async(guidance_to_speak)
            
            vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            cv2.putText(vis_frame, guidance, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Advanced Guidance", vis_frame)
            
            print(f"Frame {i+1}: {guidance}")
            
            # Stop if destination is reached
            if "Path completed" in guidance:
                print("Destination reached. Stopping guidance.")
                time.sleep(2) # Give time to see final frame
                break

            # Improved key detection
            key = cv2.waitKey(500) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping guidance")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Advanced Guidance", cv2.WND_PROP_VISIBLE) < 1:
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
                guidance = path_manager.get_guidance(frame, pose)

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
            
            # Speak the guidance asynchronously
            if not camera_covered_announced:
                speak_async(guidance_to_speak)
            
            vis_frame = path_manager.visualize_keypoints_and_matches(frame)
            cv2.putText(vis_frame, guidance, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Advanced Guidance", vis_frame)
            
            # Stop if destination is reached
            if "Path completed" in guidance:
                print("Destination reached. Stopping guidance.")
                time.sleep(2) # Give time to read message and hear audio
                break

            # Improved key detection
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), ord('x'), ord('X')]:
                print("Key pressed - stopping guidance")
                break
            # Check if window is closed
            try:
                if cv2.getWindowProperty("Advanced Guidance", cv2.WND_PROP_VISIBLE) < 1:
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
    
    path_manager = PathManager(storage_dir="demo_paths", enable_object_detection=True)
    
    # Create a few example paths with destinations
    destinations = ["kitchen", "bedroom", "office"]
    
    for i, destination in enumerate(destinations):
        print(f"\nRecording path to {destination}...")
        path_name = f"path_to_{destination}"
        
        path_manager.start_recording_path(path_name, destination)
        
        # Simulate recording (shorter for demo)
        for j in range(10):
            frame = create_test_frame(j + i * 10)
            path_manager.process_frame(frame)
            time.sleep(0.1)
        
        path_manager.stop_recording_path()
    
    print("\n✓ Created paths with destinations:")
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

def run_all_demos():
    """Run all PathManager demos."""
    print("Enhanced PathManager Features Demo with Destination Labeling")
    print("="*70)
    
    try:
        # Create demo paths directory
        os.makedirs("demo_paths", exist_ok=True)
        
        # Run demos
        demo_1_visualization()
        demo_2_advanced_guidance()
        demo_3_export_and_analysis()
        demo_4_path_management()
        demo_5_destination_labeling()
        
        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("\nGenerated files:")
        print("  - demo_export.json (path data)")
        print("  - demo_export.csv (path data)")
        print("  - demo_paths/ (recorded paths with destinations)")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_demos() 