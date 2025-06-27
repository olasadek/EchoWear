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

class EchoWearGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EchoWear Navigation System")
        self.root.geometry("900x700")
        self.root.configure(bg="#E0F7FA")  # Baby blue background
        self.path_manager = PathManager(storage_dir="demo_paths", enable_object_detection=True)
        self.cap = None
        self.running = False
        self.recording = False
        self.navigation = False
        self.frame = None
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome to EchoWear!")
        self.selected_path = None
        self.available_paths = []
        self.tts_engine = pyttsx3.init()
        self.last_spoken_guidance = None
        self.BABY_PINK = "#F8BBD0"
        self._build_ui()
        self.update_camera()

    def _build_ui(self):
        # Camera feed
        self.video_label = tk.Label(self.root, bg="#E0F7FA")
        self.video_label.pack(pady=10)

        # Controls
        controls = tk.Frame(self.root, bg="#E0F7FA")
        controls.pack(pady=5)

        tk.Label(controls, text="Path Name:", bg="#E0F7FA").grid(row=0, column=0)
        self.path_name_entry = tk.Entry(controls, width=20, bg="#E0F7FA")
        self.path_name_entry.grid(row=0, column=1)

        tk.Label(controls, text="Destination:", bg="#E0F7FA").grid(row=0, column=2)
        self.destination_entry = tk.Entry(controls, width=20, bg="#E0F7FA")
        self.destination_entry.grid(row=0, column=3)

        tk.Label(controls, text="Place Name (for geocoding):", bg="#E0F7FA").grid(row=0, column=4)
        self.place_entry = tk.Entry(controls, width=20, bg="#E0F7FA")
        self.place_entry.grid(row=0, column=5)

        self.record_btn = tk.Button(controls, text="Record Path", command=self.start_recording, bg=self.BABY_PINK)
        self.record_btn.grid(row=1, column=0, columnspan=2, pady=5)
        self.stop_record_btn = tk.Button(controls, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED, bg=self.BABY_PINK)
        self.stop_record_btn.grid(row=1, column=2, columnspan=2, pady=5)
        self.refresh_btn = tk.Button(controls, text="Refresh Paths", command=self.refresh_paths, bg=self.BABY_PINK)
        self.refresh_btn.grid(row=1, column=4)

        # Path selection
        path_frame = tk.Frame(self.root, bg="#E0F7FA")
        path_frame.pack(pady=5)
        tk.Label(path_frame, text="Available Paths:", bg="#E0F7FA").pack()
        self.path_listbox = tk.Listbox(path_frame, width=60, height=5, bg="#E0F7FA")
        self.path_listbox.pack()
        self.path_listbox.bind('<<ListboxSelect>>', self.on_path_select)
        self.refresh_paths()

        self.navigate_btn = tk.Button(self.root, text="Start Navigation", command=self.start_navigation, bg=self.BABY_PINK)
        self.navigate_btn.pack(pady=5)
        self.stop_nav_btn = tk.Button(self.root, text="Stop Navigation", command=self.stop_navigation, state=tk.DISABLED, bg=self.BABY_PINK)
        self.stop_nav_btn.pack(pady=5)

        # Status area
        self.status_label = tk.Label(self.root, textvariable=self.status_var, fg="blue", font=("Arial", 12), bg="#E0F7FA")
        self.status_label.pack(pady=10)

        self.quit_btn = tk.Button(self.root, text="Quit", command=self.quit, bg=self.BABY_PINK)
        self.quit_btn.pack(pady=5)

    def update_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read() if self.cap else (False, None)
        if ret:
            self.frame = frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        if self.running or self.recording or self.navigation:
            self.root.after(30, self.update_camera)
        else:
            self.root.after(100, self.update_camera)

    def start_recording(self):
        path_name = self.path_name_entry.get().strip()
        destination = self.destination_entry.get().strip()
        place = self.place_entry.get().strip()
        if not path_name:
            messagebox.showerror("Error", "Please enter a path name.")
            return
        coordinates = None
        if place:
            coordinates = get_coordinates_from_place(place)
            if coordinates:
                self.status_var.set(f"Geocoded '{place}': {coordinates}")
            else:
                self.status_var.set(f"Could not geocode '{place}'. Proceeding without coordinates.")
        self.path_manager.start_recording_path(path_name, destination, coordinates)
        self.recording = True
        self.running = True
        self.record_btn.config(state=tk.DISABLED)
        self.stop_record_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Recording path '{path_name}'...")
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        while self.recording:
            if self.frame is not None:
                self.path_manager.process_frame(self.frame)
            time.sleep(0.1)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.path_manager.stop_recording_path()
            self.status_var.set("Recording stopped and path saved.")
            self.record_btn.config(state=tk.NORMAL)
            self.stop_record_btn.config(state=tk.DISABLED)
            self.refresh_paths()

    def refresh_paths(self):
        self.available_paths = self.path_manager.list_paths_with_destinations()
        self.path_listbox.delete(0, tk.END)
        for info in self.available_paths:
            dest = info['destination'] or "No destination"
            self.path_listbox.insert(tk.END, f"{info['name']} → {dest}")

    def on_path_select(self, event):
        selection = self.path_listbox.curselection()
        if selection:
            self.selected_path = self.available_paths[selection[0]]['name']
        else:
            self.selected_path = None

    def start_navigation(self):
        if not self.selected_path:
            messagebox.showerror("Error", "Please select a path to navigate.")
            return
        if not self.path_manager.load_path(self.selected_path):
            messagebox.showerror("Error", f"Failed to load path '{self.selected_path}'.")
            return
        self.path_manager.start_navigation(self.selected_path)
        self.navigation = True
        self.running = True
        self.navigate_btn.config(state=tk.DISABLED)
        self.stop_nav_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Navigating path '{self.selected_path}'...")
        threading.Thread(target=self._navigation_loop, daemon=True).start()

    def _navigation_loop(self):
        while self.navigation:
            if self.frame is not None:
                mean_brightness = np.mean(self.frame)
                if mean_brightness < 15:
                    guidance = "Camera is covered."
                else:
                    pose = self.path_manager.process_frame(self.frame)
                    guidance = self.path_manager.get_guidance(self.frame, pose)
                self.status_var.set(guidance)
                if guidance != self.last_spoken_guidance:
                    self.last_spoken_guidance = guidance
                    threading.Thread(target=self.speak_guidance, args=(guidance,), daemon=True).start()
            time.sleep(0.2)

    def speak_guidance(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def stop_navigation(self):
        if self.navigation:
            self.navigation = False
            self.status_var.set("Navigation stopped.")
            self.navigate_btn.config(state=tk.NORMAL)
            self.stop_nav_btn.config(state=tk.DISABLED)

    def quit(self):
        self.running = False
        self.recording = False
        self.navigation = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EchoWearGUI(root)
    root.mainloop() 