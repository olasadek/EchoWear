#!/usr/bin/env python3
"""
Inspect Path File with Object Detection Results

Shows detailed information about recorded paths including detected objects.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from path_manager import PathManager

def inspect_path_with_objects(path_file):
    """Inspect a path file and show object detection results."""
    print(f"Analyzing path file: {path_file}")
    print("="*60)
    
    # Extract path name from filename
    path_name = os.path.basename(path_file)
    # Remove all .pkl extensions
    while path_name.endswith('.pkl'):
        path_name = path_name[:-4]
    
    # Use PathManager to get summary
    path_manager = PathManager(storage_dir=os.path.dirname(path_file))
    summary = path_manager.get_path_summary(path_name)
    
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return
    
    # Display summary
    print(f"Path Name: {summary['path_name']}")
    print(f"Total Frames: {summary['total_frames']}")
    print(f"Frames with Detections: {summary['frames_with_detections']}")
    print(f"Detection Rate: {summary['detection_rate']:.2%}")
    
    print("\nDetected Objects:")
    print("-" * 30)
    if summary['object_counts']:
        # Sort by count (most frequent first)
        sorted_objects = sorted(summary['object_counts'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for obj_name, count in sorted_objects:
            percentage = (count / summary['total_frames']) * 100
            print(f"  {obj_name}: {count} detections ({percentage:.1f}% of frames)")
    else:
        print("  No objects detected in this path")
    
    print("\n" + "="*60)

def list_all_paths(paths_dir="paths"):
    """List all available path files."""
    if not os.path.exists(paths_dir):
        print(f"Paths directory '{paths_dir}' not found.")
        return []
    
    path_files = [f for f in os.listdir(paths_dir) if f.endswith('.pkl')]
    return [os.path.join(paths_dir, f) for f in path_files]

def main():
    if len(sys.argv) > 1:
        # Inspect specific path file
        path_file = sys.argv[1]
        if not os.path.exists(path_file):
            print(f"Path file '{path_file}' not found.")
            return
        inspect_path_with_objects(path_file)
    else:
        # List and inspect all available paths
        print("Available Path Files:")
        print("="*60)
        
        path_files = list_all_paths()
        if not path_files:
            print("No path files found in 'paths' directory.")
            return
        
        for i, path_file in enumerate(path_files, 1):
            print(f"{i}. {os.path.basename(path_file)}")
        
        print(f"\nFound {len(path_files)} path file(s).")
        
        # Inspect each path
        for path_file in path_files:
            print("\n")
            inspect_path_with_objects(path_file)

if __name__ == "__main__":
    main() 