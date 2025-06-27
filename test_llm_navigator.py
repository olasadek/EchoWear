#!/usr/bin/env python3
"""
Test Script for LLM Navigator

Tests the LLM navigator with simulated scene data.
"""

import sys
import time
import logging
from pathlib import Path
import threading
import queue

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from llm_navigator import LLMNavigator, NavigationInstruction, NavigationCommand

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_detection_result():
    """Create a mock detection result for testing."""
    from dataclasses import dataclass
    
    @dataclass
    class MockDetection:
        label: str
        confidence: float
        bbox: tuple
    
    @dataclass
    class MockDetectionResult:
        detections: list
    
    # Create mock detections
    detections = [
        MockDetection("person", 0.85, (300, 200, 100, 200)),  # Person in center
        MockDetection("chair", 0.92, (500, 300, 80, 120)),    # Chair on right
        MockDetection("door", 0.78, (100, 150, 60, 180))      # Door on left
    ]
    
    return MockDetectionResult(detections=detections)

def test_llm_navigator():
    """Test the LLM navigator with mock data."""
    print("Testing LLM Navigator...")
    print("="*50)
    
    # Test 1: Rule-based mode (no model)
    print("\n1. Testing rule-based mode (no LLM model)...")
    navigator = LLMNavigator(model_path=None)
    
    mock_result = create_mock_detection_result()
    instruction = navigator.process_scene_data(mock_result)
    
    if instruction:
        print(f"✓ Generated instruction: {instruction.description}")
        print(f"  Command: {instruction.command.value}")
        print(f"  Urgency: {instruction.urgency}")
    else:
        print("✗ Failed to generate instruction")
    
    # Test 2: Voice interface
    print("\n2. Testing voice interface...")
    try:
        navigator.voice_interface.speak("Testing voice interface. This is a test message.")
        print("✓ Voice test completed")
    except Exception as e:
        print(f"✗ Voice test failed: {e}")
    
    # Test 3: Status
    print("\n3. Testing status...")
    status = navigator.get_status()
    print(f"✓ Status: {status}")
    
    # Test 4: LLM mode (if model available)
    model_path = "models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
    if Path(model_path).exists():
        print(f"\n4. Testing LLM mode with {model_path}...")
        try:
            llm_navigator = LLMNavigator(model_path=model_path)
            instruction = llm_navigator.process_scene_data(mock_result)
            
            if instruction:
                print(f"✓ LLM generated instruction: {instruction.description}")
            else:
                print("✗ LLM failed to generate instruction")
                
        except Exception as e:
            print(f"✗ LLM test failed: {e}")
    else:
        print(f"\n4. Skipping LLM test - model not found at {model_path}")
        print("   Download a GGUF model to test LLM functionality")
    
    print("\n" + "="*50)
    print("LLM Navigator test completed!")

def test_scene_scenarios():
    """Test different scene scenarios."""
    print("\nTesting different scene scenarios...")
    print("="*50)
    
    navigator = LLMNavigator(model_path=None)
    
    scenarios = [
        {
            "name": "Clear path",
            "detections": []
        },
        {
            "name": "Obstacle ahead",
            "detections": [
                ("chair", 0.9, (320, 240, 100, 100))  # Chair in center
            ]
        },
        {
            "name": "Person detected",
            "detections": [
                ("person", 0.85, (300, 200, 80, 160))  # Person ahead
            ]
        },
        {
            "name": "Door detected",
            "detections": [
                ("door", 0.8, (100, 150, 60, 180))  # Door on left
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        
        # Create mock result for this scenario
        from dataclasses import dataclass
        
        @dataclass
        class MockDetection:
            label: str
            confidence: float
            bbox: tuple
        
        @dataclass
        class MockDetectionResult:
            detections: list
        
        detections = [
            MockDetection(label, conf, bbox) 
            for label, conf, bbox in scenario['detections']
        ]
        
        mock_result = MockDetectionResult(detections=detections)
        
        # Process scene
        instruction = navigator.process_scene_data(mock_result)
        
        if instruction:
            print(f"  Instruction: {instruction.description}")
            print(f"  Command: {instruction.command.value}")
        else:
            print("  No instruction generated")

if __name__ == "__main__":
    try:
        test_llm_navigator()
        test_scene_scenarios()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1) 