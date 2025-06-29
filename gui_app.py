import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
from path_manager import PathManager
from geocoding_utils import get_coordinates_from_place
import time
import pyttsx3
import os
import sys

class EchoWearGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EchoWear Navigation System with Scene Graph Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#E0F7FA")  # Baby blue background
        
        # Initialize path manager with object detection
        self.path_manager = PathManager(storage_dir="demo_paths", enable_object_detection=True)
        
        # Camera and state variables
        self.cap = None
        self.running = False
        self.recording = False
        self.navigation = False
        self.frame = None
        
        # UI variables
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome to EchoWear Navigation System!")
        self.selected_path = None
        self.available_paths = []
        
        # TTS setup with better error handling
        self.tts_available = False
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.last_spoken_guidance = None
        self.last_guidance_time = 0
        
        # Scene graph setup
        self.scene_graph_available = False
        self.graph_builder = None
        self.scene_builder = None
        self.graph_store = None
        
        # Colors
        self.BABY_PINK = "#F8BBD0"
        self.LIGHT_GREEN = "#C8E6C9"
        self.LIGHT_ORANGE = "#FFCC80"
        
        # Initialize components
        self._init_tts()
        self._init_scene_graph()
        self._build_ui()
        self.update_camera()

    def _init_tts(self):
        """Initialize TTS engine with proper error handling."""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            self.tts_available = True
            print("✅ TTS engine initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize TTS engine: {e}")
            self.tts_available = False
            self.tts_engine = None

    def _init_scene_graph(self):
        """Initialize scene graph detection."""
        try:
            # Add the llm-camera-tracker directory to Python path
            camera_tracker_path = os.path.join(os.path.dirname(__file__), "llm-camera-tracker")
            if camera_tracker_path not in sys.path:
                sys.path.insert(0, camera_tracker_path)
            
            # Import scene graph components
            from scene_graph.hierarchical_graph_builder import HierarchicalGraphBuilder
            from scene_graph.graph_builder import SceneGraphBuilder
            from scene_graph.graph_store import GraphStore
            
            # Initialize scene graph components
            self.graph_builder = HierarchicalGraphBuilder()
            self.scene_builder = SceneGraphBuilder()
            self.graph_store = GraphStore()
            self.scene_graph_available = True
            print("✅ Scene graph detection initialized")
            
        except Exception as e:
            print(f"⚠️  Scene graph not available: {e}")
            self.scene_graph_available = False

    def _build_ui(self):
        """Build the enhanced user interface."""
        # Title
        title_label = tk.Label(self.root, text="EchoWear Navigation System", 
                              font=("Arial", 16, "bold"), bg="#E0F7FA", fg="#1976D2")
        title_label.pack(pady=10)

        # Main container
        main_container = tk.Frame(self.root, bg="#E0F7FA")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Camera and status
        left_panel = tk.Frame(main_container, bg="#E0F7FA")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Camera feed with enhanced display
        camera_frame = tk.Frame(left_panel, bg="#E0F7FA", relief=tk.RAISED, bd=2)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(camera_frame, text="Camera Feed", font=("Arial", 12, "bold"), 
                bg="#E0F7FA").pack(pady=5)
        
        self.video_label = tk.Label(camera_frame, bg="black", relief=tk.SUNKEN, bd=2)
        self.video_label.pack(pady=5, padx=5)

        # Status display
        status_frame = tk.Frame(left_panel, bg="#E0F7FA", relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(status_frame, text="Status", font=("Arial", 12, "bold"), 
                bg="#E0F7FA").pack(pady=5)
        
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, 
                                   fg="blue", font=("Arial", 10), bg="#E0F7FA", 
                                   wraplength=400, justify=tk.LEFT)
        self.status_label.pack(pady=5, padx=10)

        # Right panel - Controls
        right_panel = tk.Frame(main_container, bg="#E0F7FA")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Recording controls
        record_frame = tk.LabelFrame(right_panel, text="Path Recording", 
                                   font=("Arial", 11, "bold"), bg="#E0F7FA")
        record_frame.pack(fill=tk.X, pady=(0, 10))

        # Path name
        tk.Label(record_frame, text="Path Name:", bg="#E0F7FA").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.path_name_entry = tk.Entry(record_frame, width=25, bg="white")
        self.path_name_entry.grid(row=0, column=1, padx=5, pady=2)

        # Destination
        tk.Label(record_frame, text="Destination:", bg="#E0F7FA").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.destination_entry = tk.Entry(record_frame, width=25, bg="white")
        self.destination_entry.grid(row=1, column=1, padx=5, pady=2)

        # Place name for geocoding
        tk.Label(record_frame, text="Place Name (geocoding):", bg="#E0F7FA").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.place_entry = tk.Entry(record_frame, width=25, bg="white")
        self.place_entry.grid(row=2, column=1, padx=5, pady=2)

        # Recording buttons
        button_frame = tk.Frame(record_frame, bg="#E0F7FA")
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        self.record_btn = tk.Button(button_frame, text="🎥 Start Recording", 
                                  command=self.start_recording, bg=self.LIGHT_GREEN, 
                                  font=("Arial", 10, "bold"), width=15)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.stop_record_btn = tk.Button(button_frame, text="⏹️ Stop Recording", 
                                       command=self.stop_recording, state=tk.DISABLED, 
                                       bg=self.LIGHT_ORANGE, font=("Arial", 10, "bold"), width=15)
        self.stop_record_btn.pack(side=tk.LEFT, padx=5)

        # Navigation controls
        nav_frame = tk.LabelFrame(right_panel, text="Path Navigation", 
                                font=("Arial", 11, "bold"), bg="#E0F7FA")
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        # Path selection
        tk.Label(nav_frame, text="Available Paths:", bg="#E0F7FA").pack(anchor=tk.W, padx=5, pady=2)
        
        # Scrollable path list
        path_list_frame = tk.Frame(nav_frame, bg="#E0F7FA")
        path_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.path_listbox = tk.Listbox(path_list_frame, width=35, height=6, bg="white", 
                                     font=("Arial", 9))
        path_scrollbar = tk.Scrollbar(path_list_frame, orient=tk.VERTICAL)
        self.path_listbox.config(yscrollcommand=path_scrollbar.set)
        path_scrollbar.config(command=self.path_listbox.yview)
        
        self.path_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        path_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.path_listbox.bind('<<ListboxSelect>>', self.on_path_select)

        # Navigation buttons
        nav_button_frame = tk.Frame(nav_frame, bg="#E0F7FA")
        nav_button_frame.pack(fill=tk.X, pady=10)

        self.refresh_btn = tk.Button(nav_button_frame, text="🔄 Refresh Paths", 
                                   command=self.refresh_paths, bg=self.BABY_PINK, 
                                   font=("Arial", 9, "bold"))
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        self.navigate_btn = tk.Button(nav_button_frame, text="🧭 Start Navigation", 
                                    command=self.start_navigation, bg=self.LIGHT_GREEN, 
                                    font=("Arial", 9, "bold"))
        self.navigate_btn.pack(side=tk.LEFT, padx=5)

        self.stop_nav_btn = tk.Button(nav_button_frame, text="⏹️ Stop Navigation", 
                                    command=self.stop_navigation, state=tk.DISABLED, 
                                    bg=self.LIGHT_ORANGE, font=("Arial", 9, "bold"))
        self.stop_nav_btn.pack(side=tk.LEFT, padx=5)

        # GPS Navigation button
        self.gps_nav_btn = tk.Button(nav_button_frame, text="🌍 GPS Navigation", 
                                    command=self.open_gps_navigation, bg="#B3E5FC", 
                                    font=("Arial", 9, "bold"))
        self.gps_nav_btn.pack(side=tk.LEFT, padx=5)

        # System info
        info_frame = tk.LabelFrame(right_panel, text="System Status", 
                                 font=("Arial", 11, "bold"), bg="#E0F7FA")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # TTS status
        tts_status = "✅ Voice Guidance Active" if self.tts_available else "❌ Voice Guidance Disabled"
        tk.Label(info_frame, text=f"TTS: {tts_status}", bg="#E0F7FA", 
                fg="green" if self.tts_available else "red").pack(anchor=tk.W, padx=5, pady=2)

        # Scene graph status
        sg_status = "✅ Scene Graph Active" if self.scene_graph_available else "❌ Scene Graph Disabled"
        tk.Label(info_frame, text=f"Scene Graph: {sg_status}", bg="#E0F7FA", 
                fg="green" if self.scene_graph_available else "red").pack(anchor=tk.W, padx=5, pady=2)

        # Camera status
        self.camera_status_var = tk.StringVar(value="📷 Camera: Initializing...")
        tk.Label(info_frame, textvariable=self.camera_status_var, bg="#E0F7FA").pack(anchor=tk.W, padx=5, pady=2)

        # Camera reset button
        self.reset_camera_btn = tk.Button(info_frame, text="🔄 Reset Camera", 
                                        command=self.reset_camera, bg="#FFE0B2", 
                                        font=("Arial", 8, "bold"))
        self.reset_camera_btn.pack(anchor=tk.W, padx=5, pady=2)

        # Quit button
        self.quit_btn = tk.Button(right_panel, text="🚪 Quit", command=self.quit, 
                                bg="#FFCDD2", font=("Arial", 11, "bold"), width=15)
        self.quit_btn.pack(pady=10)

        # Initialize paths
        self.refresh_paths()

    def update_camera(self):
        """Update camera feed with enhanced error handling."""
        if self.cap is None:
            # Try to initialize camera with better error handling
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    # Try alternative camera index
                    self.cap = cv2.VideoCapture(1)
                    if not self.cap.isOpened():
                        self.cap = None
                        self.camera_status_var.set("❌ Camera: Not Available")
                        # Create a placeholder image
                        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Camera Not Available", (150, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(placeholder, "Check if camera is in use", (120, 220), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        self.frame = placeholder
                    else:
                        self.camera_status_var.set("✅ Camera: Active (Index 1)")
                else:
                    self.camera_status_var.set("✅ Camera: Active (Index 0)")
            except Exception as e:
                print(f"Camera initialization error: {e}")
                self.cap = None
                self.camera_status_var.set("❌ Camera: Initialization Failed")
                # Create a placeholder image
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Error", (200, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.frame = placeholder
        
        # Try to read frame with error handling
        frame_retrieved = False
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame = frame
                    frame_retrieved = True
                else:
                    # Frame grab failed, try to reinitialize camera
                    print("Frame grab failed, attempting to reinitialize camera...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.1)  # Brief pause before retry
            except Exception as e:
                print(f"Frame read error: {e}")
                # Release and reset camera on error
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        # Display frame or placeholder
        if frame_retrieved and self.frame is not None:
            # Add scene graph overlay if available
            if self.scene_graph_available and (self.recording or self.navigation):
                self.frame = self._add_scene_graph_overlay(self.frame)
            
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            # Show placeholder or last frame
            if not hasattr(self, 'last_good_frame'):
                # Create a placeholder image
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Unavailable", (150, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(placeholder, "Try closing other camera apps", (120, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.last_good_frame = placeholder
            
            img = Image.fromarray(self.last_good_frame)
            img = img.resize((640, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Update frequency based on activity
        if self.running or self.recording or self.navigation:
            self.root.after(30, self.update_camera)  # 30ms for active mode
        else:
            self.root.after(100, self.update_camera)  # 100ms for idle mode

    def _add_scene_graph_overlay(self, frame):
        """Add scene graph information overlay to the frame."""
        try:
            if self.path_manager.enable_object_detection and self.path_manager.object_detector:
                detection_result = self.path_manager.object_detector.detect(frame)
                if detection_result and detection_result.detections:
                    detected_objects = []
                    for detection in detection_result.detections:
                        if detection.confidence > 0.5:
                            detected_objects.append(detection.label)
                    
                    if detected_objects:
                        # Create scene description
                        if len(detected_objects) == 1:
                            scene_description = f"There is a {detected_objects[0]} in the scene."
                        else:
                            scene_description = f"There are {', '.join(detected_objects[:-1])} and {detected_objects[-1]} in the scene."
                        
                        # Update scene graph
                        if self.graph_builder:
                            try:
                                scene_graph, action_graph, object_graph = self.graph_builder.update_scene_state(
                                    scene_description, time.time()
                                )
                            except Exception as e:
                                print(f"Scene graph update error: {e}")
                        
                        # Add overlay text
                        cv2.putText(frame, "Scene Graph Active", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Scene: {scene_description}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Add detection boxes
                        for detection in detection_result.detections:
                            if detection.confidence > 0.5:
                                x, y, w, h = detection.bbox
                                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{detection.label} ({detection.confidence:.2f})", 
                                           (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception as e:
            print(f"Scene graph overlay error: {e}")
        
        return frame

    def start_recording(self):
        """Start recording a new path with enhanced validation."""
        path_name = self.path_name_entry.get().strip()
        destination = self.destination_entry.get().strip()
        place = self.place_entry.get().strip()
        
        if not path_name:
            messagebox.showerror("Error", "Please enter a path name.")
            return
        
        # Get coordinates if place name provided
        coordinates = None
        if place:
            try:
                coordinates = get_coordinates_from_place(place)
                if coordinates:
                    self.status_var.set(f"✅ Geocoded '{place}': {coordinates}")
                else:
                    self.status_var.set(f"⚠️ Could not geocode '{place}'. Proceeding without coordinates.")
            except Exception as e:
                self.status_var.set(f"⚠️ Geocoding error: {e}")
        
        # Start recording
        try:
            self.path_manager.start_recording_path(path_name, destination, coordinates)
            self.recording = True
            self.running = True
            self.record_btn.config(state=tk.DISABLED)
            self.stop_record_btn.config(state=tk.NORMAL)
            self.status_var.set(f"🎥 Recording path '{path_name}'...")
            
            # Welcome message with TTS
            if self.tts_available:
                welcome_msg = f"Starting to record path to {destination or 'unknown destination'}"
                self._speak_async(welcome_msg)
            
            threading.Thread(target=self._record_loop, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.status_var.set(f"❌ Recording failed: {e}")

    def _record_loop(self):
        """Recording loop with scene graph integration."""
        while self.recording:
            if self.frame is not None:
                try:
                    self.path_manager.process_frame(self.frame)
                except Exception as e:
                    print(f"Frame processing error: {e}")
            time.sleep(0.1)

    def stop_recording(self):
        """Stop recording and save the path."""
        if self.recording:
            try:
                self.recording = False
                self.path_manager.stop_recording_path()
                self.status_var.set("✅ Recording stopped and path saved.")
                
                # Completion message with TTS
                if self.tts_available:
                    self._speak_async("Path recording completed")
                
                self.record_btn.config(state=tk.NORMAL)
                self.stop_record_btn.config(state=tk.DISABLED)
                self.refresh_paths()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop recording: {e}")
                self.status_var.set(f"❌ Error stopping recording: {e}")

    def refresh_paths(self):
        """Refresh the list of available paths."""
        try:
            self.available_paths = self.path_manager.list_paths_with_destinations()
            self.path_listbox.delete(0, tk.END)
            for info in self.available_paths:
                dest = info['destination'] or "No destination"
                self.path_listbox.insert(tk.END, f"{info['name']} → {dest}")
            
            if self.available_paths:
                self.status_var.set(f"📁 Found {len(self.available_paths)} saved paths")
            else:
                self.status_var.set("📁 No saved paths found")
                
        except Exception as e:
            self.status_var.set(f"❌ Error refreshing paths: {e}")

    def on_path_select(self, event):
        """Handle path selection."""
        selection = self.path_listbox.curselection()
        if selection:
            self.selected_path = self.available_paths[selection[0]]['name']
            dest = self.available_paths[selection[0]]['destination'] or "No destination"
            self.status_var.set(f"📋 Selected: {self.selected_path} → {dest}")
        else:
            self.selected_path = None

    def start_navigation(self):
        """Start navigation with enhanced error handling."""
        if not self.selected_path:
            messagebox.showerror("Error", "Please select a path to navigate.")
            return
        
        try:
            if not self.path_manager.load_path(self.selected_path):
                messagebox.showerror("Error", f"Failed to load path '{self.selected_path}'.")
                return
            
            self.path_manager.start_navigation(self.selected_path)
            self.navigation = True
            self.running = True
            self.navigate_btn.config(state=tk.DISABLED)
            self.stop_nav_btn.config(state=tk.NORMAL)
            
            # Welcome message
            dest = next((p['destination'] for p in self.available_paths if p['name'] == self.selected_path), "unknown destination")
            welcome_msg = f"Starting navigation to {dest}. Voice guidance is active."
            self.status_var.set(welcome_msg)
            
            if self.tts_available:
                self._speak_async(welcome_msg)
            
            threading.Thread(target=self._navigation_loop, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start navigation: {e}")
            self.status_var.set(f"❌ Navigation failed: {e}")

    def _navigation_loop(self):
        """Navigation loop with scene graph and TTS integration."""
        last_guidance_time = 0
        
        while self.navigation:
            if self.frame is not None:
                try:
                    # Check camera coverage
                    mean_brightness = np.mean(self.frame)
                    if mean_brightness < 15:
                        guidance = "Camera is covered."
                    else:
                        pose = self.path_manager.process_frame(self.frame)
                        guidance = self.path_manager.get_guidance(self.frame, pose)
                    
                    self.status_var.set(guidance)
                    
                    # Smart TTS - only speak if guidance changed and enough time passed
                    current_time = time.time()
                    if (guidance != self.last_spoken_guidance and 
                        current_time - last_guidance_time > 3.0):
                        self.last_spoken_guidance = guidance
                        last_guidance_time = current_time
                        self._speak_async(guidance)
                    
                    # Check if destination reached
                    if "Path completed" in guidance:
                        completion_msg = "Destination reached. Navigation complete."
                        self.status_var.set(completion_msg)
                        if self.tts_available:
                            self._speak_async(completion_msg)
                        time.sleep(2)
                        self.stop_navigation()
                        break
                        
                except Exception as e:
                    print(f"Navigation error: {e}")
                    self.status_var.set(f"⚠️ Navigation error: {e}")
            
            time.sleep(0.2)

    def _speak_async(self, text):
        """Safely speak text asynchronously."""
        if self.tts_available and self.tts_engine:
            def speak():
                try:
                    with self.tts_lock:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
            
            threading.Thread(target=speak, daemon=True).start()

    def stop_navigation(self):
        """Stop navigation."""
        if self.navigation:
            self.navigation = False
            self.status_var.set("⏹️ Navigation stopped.")
            self.navigate_btn.config(state=tk.NORMAL)
            self.stop_nav_btn.config(state=tk.DISABLED)

    def quit(self):
        """Clean shutdown of the application."""
        self.running = False
        self.recording = False
        self.navigation = False
        
        # Properly release camera
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Camera release error: {e}")
            self.cap = None
        
        # Stop TTS engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                print(f"TTS stop error: {e}")
        
        # Destroy the window
        self.root.destroy()

    def open_gps_navigation(self):
        """Open the GPS Navigation window (Demo 6 workflow)."""
        GPSNavigationWindow(self.root)

    def reset_camera(self):
        """Reset the camera."""
        self.cap = None
        self.camera_status_var.set("📷 Camera: Resetting...")
        self.update_camera()

class GPSNavigationWindow:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("GPS Navigation - EchoWear")
        self.top.geometry("900x700")
        self.top.configure(bg="#E0F7FA")
        self.status_var = tk.StringVar(value="Initializing GPS navigation...")
        self.place_var = tk.StringVar()
        self.places = []
        self.selected_place = None
        self.cap = None
        self.running = False
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.last_guidance = None
        self._init_tts()
        self._build_ui()
        threading.Thread(target=self._init_location, daemon=True).start()

    def _init_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
        except Exception as e:
            self.tts_engine = None
            print(f"TTS error: {e}")

    def _speak_async(self, text):
        if self.tts_engine:
            def speak():
                try:
                    with self.tts_lock:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
            threading.Thread(target=speak, daemon=True).start()

    def _build_ui(self):
        tk.Label(self.top, text="GPS Navigation", font=("Arial", 16, "bold"), bg="#E0F7FA", fg="#1976D2").pack(pady=10)
        status_frame = tk.Frame(self.top, bg="#E0F7FA")
        status_frame.pack(pady=5)
        tk.Label(status_frame, textvariable=self.status_var, fg="blue", font=("Arial", 11), bg="#E0F7FA").pack()
        
        search_frame = tk.Frame(self.top, bg="#E0F7FA")
        search_frame.pack(pady=5)
        tk.Label(search_frame, text="Search for place type (e.g. cafe, hospital, park):", bg="#E0F7FA").pack(side=tk.LEFT)
        self.query_entry = tk.Entry(search_frame, textvariable=self.place_var, width=20)
        self.query_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="Search Nearby", command=self.search_places, bg="#B3E5FC").pack(side=tk.LEFT, padx=5)
        
        self.places_listbox = tk.Listbox(self.top, width=60, height=6, bg="white", font=("Arial", 10))
        self.places_listbox.pack(pady=5)
        self.places_listbox.bind('<<ListboxSelect>>', self.on_place_select)
        
        tk.Button(self.top, text="Start GPS Navigation", command=self.start_gps_navigation, bg="#C8E6C9", font=("Arial", 11, "bold")).pack(pady=10)
        
        self.video_label = tk.Label(self.top, bg="black", relief=tk.SUNKEN, bd=2)
        self.video_label.pack(pady=10)
        tk.Button(self.top, text="Close", command=self.close, bg="#FFCDD2", font=("Arial", 10, "bold")).pack(pady=5)

    def _init_location(self):
        import geocoder
        self.status_var.set("Detecting current location...")
        try:
            g = geocoder.ip('me')
            if g.latlng:
                self.latitude, self.longitude = g.latlng
                self.status_var.set(f"Current location: {self.latitude:.5f}, {self.longitude:.5f}")
                self._speak_async("Location detected. Enter a place type and search.")
            else:
                self.latitude, self.longitude = 40.7580, -73.9855
                self.status_var.set("Could not detect location. Using default: New York City.")
        except Exception as e:
            self.latitude, self.longitude = 40.7580, -73.9855
            self.status_var.set(f"Location error: {e}. Using default: New York City.")

    def search_places(self):
        from geocoding_utils import find_nearby_places
        query = self.place_var.get().strip()
        self.status_var.set("Searching for nearby places...")
        self.places = find_nearby_places(self.latitude, self.longitude, query=query, distance_km=1)
        self.places_listbox.delete(0, tk.END)
        if self.places:
            for i, place in enumerate(self.places, 1):
                name = place.get('name', 'N/A')
                address = place.get('address', 'N/A')
                distance = place.get('distance_km', 'N/A')
                self.places_listbox.insert(tk.END, f"{i}. {name} ({distance} km) - {address}")
            self.status_var.set(f"Found {len(self.places)} places. Select one to navigate.")
        else:
            self.status_var.set("No places found.")

    def on_place_select(self, event):
        selection = self.places_listbox.curselection()
        if selection:
            self.selected_place = self.places[selection[0]]
        else:
            self.selected_place = None

    def start_gps_navigation(self):
        if not self.selected_place:
            self.status_var.set("Please select a place to navigate to.")
            return
        self.status_var.set("Starting GPS navigation...")
        self._speak_async(f"Starting navigation to {self.selected_place.get('name', 'the selected place')}")
        self.running = True
        threading.Thread(target=self._gps_navigation_loop, daemon=True).start()

    def _gps_navigation_loop(self):
        import geocoder
        import cv2
        import numpy as np
        import time
        target_coords = self.selected_place.get('coordinates', None)
        target_name = self.selected_place.get('name', 'Unknown location')
        if not target_coords:
            self.status_var.set("Selected place has no coordinates.")
            return
        target_lat, target_lon = target_coords
        
        # Initialize camera with better error handling
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try alternative camera index
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.status_var.set("Camera not available for GPS navigation.")
                return
        
        self.status_var.set(f"Navigating to {target_name}...")
        last_guidance = None
        frame_count = 0
        
        while self.running:
            # Get current location
            try:
                g = geocoder.ip('me')
                if g.latlng:
                    current_lat, current_lon = g.latlng
                else:
                    self.status_var.set("Lost GPS signal.")
                    time.sleep(2)
                    continue
            except Exception as e:
                self.status_var.set(f"GPS error: {e}")
                time.sleep(2)
                continue
            
            # Calculate distance and bearing
            from enhanced_path_manager_demo import calculate_distance_and_bearing, get_cardinal_direction
            distance, bearing = calculate_distance_and_bearing(current_lat, current_lon, target_lat, target_lon)
            
            # Camera frame with error handling
            frame = None
            if self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        # Frame grab failed, try to reinitialize
                        print("GPS Navigation: Frame grab failed, reinitializing camera...")
                        self.cap.release()
                        time.sleep(0.5)
                        self.cap = cv2.VideoCapture(0)
                        if not self.cap.isOpened():
                            self.cap = cv2.VideoCapture(1)
                        if self.cap.isOpened():
                            ret, frame = self.cap.read()
                except Exception as e:
                    print(f"GPS Navigation camera error: {e}")
                    frame = None
            
            # Create frame if camera failed
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Unavailable", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Obstacle detection (simple edge/contour analysis)
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                obstacle_detected = False
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:
                        obstacle_detected = True
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Obstacle ({area:.0f})", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except Exception as e:
                print(f"Obstacle detection error: {e}")
                obstacle_detected = False
            
            # Overlay info
            cv2.putText(frame, f"Distance: {int(distance)}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Direction: {get_cardinal_direction(bearing)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Target: {target_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if obstacle_detected:
                cv2.putText(frame, "OBSTACLE DETECTED!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Show frame in Tkinter
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((640, 360))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            except Exception as e:
                print(f"Frame display error: {e}")
            
            # Guidance
            guidance = f"Head {get_cardinal_direction(bearing)}, {int(distance)} meters to {target_name}"
            if obstacle_detected:
                guidance = "Obstacle detected ahead. " + guidance
            
            self.status_var.set(guidance)
            
            # Smart TTS - only speak if guidance changed and enough time passed
            if guidance != last_guidance:
                self._speak_async(guidance)
                last_guidance = guidance
            
            # Check if arrived
            if distance < 20:
                self.status_var.set(f"Arrived at {target_name}!")
                self._speak_async(f"You have arrived at {target_name}!")
                break
            
            # Update every second, but check for window close more frequently
            for _ in range(10):  # Check 10 times per second
                if not self.running:
                    break
                time.sleep(0.1)
        
        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None

    def close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.top.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EchoWearGUI(root)
    root.mainloop() 