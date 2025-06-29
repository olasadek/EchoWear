"""
LLM Navigator Module for EchoWear

This module integrates scene graph data from the camera processing pipeline
with LLM-based scene interpretation and voice guidance for wearable devices.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)

class NavigationCommand(Enum):
    """Available navigation commands."""
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    CAUTION = "caution"
    CLEAR_PATH = "clear_path"
    OBSTACLE_AHEAD = "obstacle_ahead"
    DOOR_DETECTED = "door_detected"
    STAIRS_DETECTED = "stairs_detected"

@dataclass
class SceneObject:
    """Represents a detected object in the scene."""
    label: str
    confidence: float
    distance: Optional[float] = None  # Estimated distance in meters
    position: Optional[Tuple[float, float]] = None  # (x, y) relative position
    size: Optional[Tuple[float, float]] = None  # (width, height) in pixels

@dataclass
class NavigationInstruction:
    """Generated navigation instruction."""
    command: NavigationCommand
    description: str
    confidence: float
    urgency: float  # 0.0 to 1.0, higher means more urgent
    timestamp: float

class LLMInterface:
    """
    Interface for LLM-based navigation command generation using ctransformers.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize LLM interface.
        
        Args:
            model_path: Path to GGUF model file (optional, will use fallback if not provided)
        """
        self.model = None
        self.model_path = model_path
        self.last_command = None
        self.command_cooldown = 2.0  # seconds between commands
        self.last_command_time = 0
        
        if model_path:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM model using ctransformers."""
        try:
            from ctransformers import AutoModelForCausalLM
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type="llama",  # or "phi" for Phi-2 models
                gpu_layers=0,  # CPU only for compatibility
                threads=4
            )
            logger.info(f"LLM model loaded from {self.model_path}")
            
        except ImportError:
            logger.warning("ctransformers not available, using rule-based system")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
    
    def generate_navigation_command(self, scene_analysis: Dict[str, Any]) -> NavigationInstruction:
        """
        Generate navigation command based on scene analysis.
        
        Args:
            scene_analysis: Scene analysis results from SceneAnalyzer
            
        Returns:
            NavigationInstruction with command and description
        """
        # Check cooldown
        current_time = time.time()
        if (self.last_command and 
            current_time - self.last_command_time < self.command_cooldown):
            return self.last_command
        
        if self.model:
            return self._generate_llm_command(scene_analysis)
        else:
            return self._generate_rule_based_command(scene_analysis)
    
    def _generate_llm_command(self, scene_analysis: Dict[str, Any]) -> NavigationInstruction:
        """Generate command using LLM."""
        try:
            # Create scene description
            scene_description = self._create_scene_description(scene_analysis)
            
            # Create prompt for navigation
            prompt = f"""You are a navigation assistant for a visually impaired person. Based on the scene description, provide a short, clear navigation instruction.

Scene: {scene_description}

Navigation instruction:"""
            
            # Generate response
            response = self.model(
                prompt,
                max_new_tokens=50,
                temperature=0.7,
                stop=["\n\n", "Scene:", "Navigation:"]
            )
            
            # Extract the instruction
            instruction_text = response.strip()
            
            # Map to navigation command
            command, urgency = self._map_instruction_to_command(instruction_text, scene_analysis)
            
            return NavigationInstruction(
                command=command,
                description=instruction_text,
                confidence=0.8,
                urgency=urgency,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_rule_based_command(scene_analysis)
    
    def _create_scene_description(self, scene_analysis: Dict[str, Any]) -> str:
        """Create a natural language description of the scene."""
        objects = scene_analysis.get('objects', [])
        
        if not objects:
            return "No objects detected in the scene."
        
        descriptions = []
        for obj in objects:
            distance_str = f"{obj['distance']:.1f}m" if obj.get('distance') else "unknown distance"
            descriptions.append(f"{obj['label']} at {distance_str}")
        
        return f"Detected: {', '.join(descriptions)}."
    
    def _map_instruction_to_command(self, instruction: str, scene_analysis: Dict[str, Any]) -> Tuple[NavigationCommand, float]:
        """Map LLM instruction to navigation command."""
        instruction_lower = instruction.lower()
        
        # Simple keyword mapping
        if any(word in instruction_lower for word in ['stop', 'halt', 'danger']):
            return NavigationCommand.STOP, 0.9
        elif any(word in instruction_lower for word in ['obstacle', 'blocked', 'blocking']):
            return NavigationCommand.OBSTACLE_AHEAD, 0.7
        elif any(word in instruction_lower for word in ['door', 'entrance']):
            return NavigationCommand.DOOR_DETECTED, 0.6
        elif any(word in instruction_lower for word in ['stairs', 'steps']):
            return NavigationCommand.STAIRS_DETECTED, 0.8
        elif any(word in instruction_lower for word in ['caution', 'careful', 'slow']):
            return NavigationCommand.CAUTION, 0.5
        elif any(word in instruction_lower for word in ['clear', 'safe', 'forward']):
            return NavigationCommand.CLEAR_PATH, 0.1
        else:
            return NavigationCommand.CAUTION, 0.3
    
    def _generate_rule_based_command(self, scene_analysis: Dict[str, Any]) -> NavigationInstruction:
        """Generate command using rule-based logic (fallback)."""
        objects = scene_analysis.get('objects', [])
        hazards = scene_analysis.get('hazards', [])
        obstacles = scene_analysis.get('obstacles', [])
        navigation_points = scene_analysis.get('navigation_points', [])
        spatial_analysis = scene_analysis.get('spatial_analysis', {})
        
        # Check for immediate hazards
        if hazards:
            closest_hazard = min(hazards, key=lambda x: x.get('distance', float('inf')))
            if closest_hazard.get('distance', 0) < 2.5:  # Increased from 1.5 for more flexibility
                return NavigationInstruction(
                    command=NavigationCommand.STOP,
                    description=f"Stop! {closest_hazard['label']} detected ahead",
                    confidence=closest_hazard['confidence'],
                    urgency=0.9,
                    timestamp=time.time()
                )
        
        # Check for obstacles
        if not spatial_analysis.get('path_clear', True):
            closest_obstacle = spatial_analysis.get('closest_obstacle')
            if closest_obstacle:
                distance = closest_obstacle.get('distance', 0)
                if distance < 1.5:  # Increased from 1.0 for more flexibility
                    return NavigationInstruction(
                        command=NavigationCommand.OBSTACLE_AHEAD,
                        description=f"Obstacle ahead: {closest_obstacle['label']}",
                        confidence=closest_obstacle['confidence'],
                        urgency=0.7,
                        timestamp=time.time()
                    )
        
        # Default: path is clear
        return NavigationInstruction(
            command=NavigationCommand.CLEAR_PATH,
            description="Path is clear, continue forward",
            confidence=0.8,
            urgency=0.1,
            timestamp=time.time()
        )

class VoiceInterface:
    """
    Text-to-Speech interface for vocalizing navigation commands.
    """
    
    def __init__(self, use_local_tts: bool = True):
        """
        Initialize voice interface.
        
        Args:
            use_local_tts: Whether to use local TTS (True) or cloud TTS (False)
        """
        self.use_local_tts = use_local_tts
        self.tts_engine = None
        self.voice_queue = queue.Queue()
        self.voice_thread = None
        self.is_speaking = False
        
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize TTS engine."""
        try:
            if self.use_local_tts:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                # Configure voice settings
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 0.8)  # Volume
                logger.info("Local TTS engine initialized")
            else:
                # TODO: Implement cloud TTS (gTTS, Azure, etc.)
                logger.info("Cloud TTS not implemented, falling back to local TTS")
                self.use_local_tts = True
                self._initialize_tts()
                
        except ImportError:
            logger.warning("pyttsx3 not available, voice commands will be text-only")
            self.tts_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_engine = None
    
    def speak(self, text: str, priority: bool = False):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            priority: Whether this is a priority message (interrupts current speech)
        """
        if priority and self.is_speaking:
            self.stop_speaking()
        
        if self.tts_engine:
            self.voice_queue.put(text)
            if not self.voice_thread or not self.voice_thread.is_alive():
                self.voice_thread = threading.Thread(target=self._speak_worker, daemon=True)
                self.voice_thread.start()
        else:
            # Fallback: print to console
            print(f"[VOICE] {text}")
    
    def _speak_worker(self):
        """Worker thread for TTS processing."""
        while True:
            try:
                text = self.voice_queue.get(timeout=1.0)
                if text is None:  # Shutdown signal
                    break
                
                self.is_speaking = True
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except RuntimeError as e:
                    if "run loop already started" in str(e):
                        # Ignore and continue
                        pass
                    else:
                        raise
                self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error: {e}")
                self.is_speaking = False
    
    def stop_speaking(self):
        """Stop current speech."""
        if self.tts_engine and self.is_speaking:
            self.tts_engine.stop()
            self.is_speaking = False

class LLMNavigator:
    """
    Main navigator class that integrates scene analysis, LLM processing, and voice guidance.
    """
    
    def __init__(self, model_path: str = None, use_local_tts: bool = True):
        """
        Initialize the LLM navigator.
        
        Args:
            model_path: Path to GGUF model file (optional)
            use_local_tts: Whether to use local TTS for voice output
        """
        self.llm_interface = LLMInterface(model_path=model_path)
        self.voice_interface = VoiceInterface(use_local_tts=use_local_tts)
        
        # Processing state
        self.is_running = False
        self.last_instruction = None
        self.instruction_count = 0
        
        logger.info("LLM Navigator initialized")
    
    def process_scene_data(self, detection_result) -> Optional[NavigationInstruction]:
        """
        Process scene data from camera pipeline and generate navigation instruction.
        
        Args:
            detection_result: Detection result from camera processing pipeline
            
        Returns:
            NavigationInstruction if generated, None otherwise
        """
        try:
            # Support both real and mock detection results
            if hasattr(detection_result, 'detection_result'):
                detections = detection_result.detection_result.detections
            else:
                detections = detection_result.detections
            
            # Create scene analysis
            scene_analysis = self._analyze_scene(detections)
            
            # Generate navigation command
            instruction = self.llm_interface.generate_navigation_command(scene_analysis)
            
            # Update state
            self.last_instruction = instruction
            self.instruction_count += 1
            
            # Speak the instruction
            self._speak_instruction(instruction)
            
            return instruction
            
        except Exception as e:
            logger.error(f"Error processing scene data: {e}")
            return None
    
    def _analyze_scene(self, detections: List[Any]) -> Dict[str, Any]:
        """Analyze detected objects and generate scene understanding."""
        scene_objects = []
        hazards = []
        obstacles = []
        navigation_points = []
        
        # Hazard categories
        hazard_objects = {
            'person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
            'stairs', 'fire hydrant', 'stop sign', 'traffic light'
        }
        
        # Obstacle objects
        obstacle_objects = {
            'chair', 'table', 'couch', 'bed', 'door', 'wall', 'tree',
            'fence', 'building', 'bench', 'pole'
        }
        
        # Navigation objects
        navigation_objects = {
            'door', 'stairs', 'elevator', 'exit', 'entrance'
        }
        
        for detection in detections:
            # Estimate distance (rough calculation)
            distance = self._estimate_distance(detection.bbox, detection.label)
            
            scene_obj = {
                'label': detection.label,
                'confidence': detection.confidence,
                'distance': distance,
                'bbox': detection.bbox
            }
            scene_objects.append(scene_obj)
            
            # Categorize objects
            if detection.label in hazard_objects:
                hazards.append(scene_obj)
            elif detection.label in obstacle_objects:
                obstacles.append(scene_obj)
            elif detection.label in navigation_objects:
                navigation_points.append(scene_obj)
        
        return {
            'objects': scene_objects,
            'hazards': hazards,
            'obstacles': obstacles,
            'navigation_points': navigation_points,
            'spatial_analysis': {
                'path_clear': not any(obj['distance'] < 3.0 for obj in obstacles if obj['distance'])
            }
        }
    
    def _estimate_distance(self, bbox: Tuple[int, int, int, int], label: str) -> float:
        """Rough distance estimation based on bounding box size."""
        x, y, w, h = bbox
        area = w * h
        
        # Known object sizes (approximate, in meters)
        known_sizes = {
            'person': 1.7,  # average height
            'chair': 0.5,   # typical height
            'table': 0.7,   # typical height
            'door': 2.0,    # typical height
            'car': 1.5,     # typical height
        }
        
        if label in known_sizes:
            # Rough distance estimation
            estimated_distance = (known_sizes[label] * 500) / max(w, h)
            return max(0.5, min(10.0, estimated_distance))
        
        # Default estimation
        return max(0.5, min(10.0, 1000 / area))
    
    def _speak_instruction(self, instruction: NavigationInstruction):
        """Speak the navigation instruction."""
        # Determine priority based on urgency
        priority = instruction.urgency > 0.7
        
        # Speak the description
        self.voice_interface.speak(instruction.description, priority=priority)
        
        # Log the instruction
        logger.info(f"Navigation: {instruction.command.value} - {instruction.description}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get navigator status."""
        return {
            'is_running': self.is_running,
            'instruction_count': self.instruction_count,
            'last_instruction': self.last_instruction.command.value if self.last_instruction else None,
            'voice_available': self.voice_interface.tts_engine is not None,
            'llm_available': self.llm_interface.model is not None
        }
    
    def stop(self):
        """Stop the navigator."""
        self.is_running = False
        if self.voice_interface.voice_thread:
            self.voice_interface.voice_queue.put(None)  # Shutdown signal