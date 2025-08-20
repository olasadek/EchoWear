#!/usr/bin/env python3
"""
Script to update enhanced_path_manager_demo.py to use local modules
"""

import re

def update_enhanced_demo():
    """Update the enhanced demo file to use local modules instead of submodule"""
    
    # Read the original file
    with open('enhanced_path_manager_demo.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to replace submodule path setup
    old_pattern = '''        # Add the llm-camera-tracker directory to Python path
        camera_tracker_path = os.path.join(os.path.dirname(__file__), "llm-camera-tracker")
        if camera_tracker_path not in sys.path:
            sys.path.insert(0, camera_tracker_path)
        
        # Import scene graph components'''
    
    new_pattern = '''        # Import scene graph components from local module'''
    
    # Replace all instances
    updated_content = content.replace(old_pattern, new_pattern)
    
    # Write the updated content
    with open('enhanced_path_manager_demo.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✓ Updated enhanced_path_manager_demo.py")
    print(f"✓ Made {content.count(old_pattern)} replacements")

if __name__ == "__main__":
    update_enhanced_demo()
