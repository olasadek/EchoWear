# EchoWear: AI Navigation Helper for the Visually Impaired

EchoWear is a prototype AI-powered navigation assistant designed to help visually impaired users navigate indoor and outdoor environments. A protoype for navigation glasses with AI ship. It uses a wearable camera, real-time computer vision, object detection, and voice guidance to provide safe, context-aware navigation instructions.
![b2683635-b3e7-4531-bdb9-9c19ccfc6c88](https://github.com/user-attachments/assets/106ea61b-35cd-4dcf-8884-50ffa628f79b)

*Attribution*:
In honor of Dr. Ammar Mohanna for his invaluable efforts to the AI community in lebanon and continuous help .

## Features
- Real-time camera feed processing
- Visual path recording and navigation
- Object detection and obstacle warnings
- Voice guidance (TTS)
- Geocoding for destination selection
- Desktop GUI (Tkinter)
- Dockerized for easy deployment
- Automated smoke and pressure tests
- **Scene graph** construction for advanced spatial understanding
- Modular architecture with a dedicated submodule for camera tracking and scene graphing

## Dependency Summary & Version Pinning
- **Python**: 3.10 (recommended; 3.8+ supported)
- **OpenCV**: >=4.8.0
- **NumPy**: 1.24.3
- **spaCy**: 3.5.3
- **thinc**: 8.1.12
- **pyttsx3**: >=2.90
- **pillow**: >=10.0.0
- **torch**: >=2.0.0 (optional, for LLM/ML features)
- **transformers**: >=4.30.0 (optional)
- **fastapi**, **uvicorn**, **aiohttp**, **networkx**, **pyvis**, **requests**, **nltk**, **scikit-learn**, **gensim**, **tokenizers**
- See `requirements.txt` for full list and pinned versions.

**System dependencies (Linux/Docker):**
- `python3-tk`, `libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxrender1`, `libxext6`, `alsa-utils`, `espeak`, `ffmpeg`, `pulseaudio`, `libespeak1`

## Sample Output / Demo

**Sample TTS Output:**
```
[VOICE] Turn left in 5 meters. Obstacle detected: chair ahead.
[VOICE] Camera is covered.
[VOICE] You have reached your destination.
```

**Sample Console Output (Pressure Test):**
```
--- EchoWear Pressure Test ---
Processing 500 dummy frames...
Frame processing done in 8.12s with 0 errors.
Calling TTS 100 times...
TTS done in 3.21s with 0 errors.
--- Pressure Test Complete ---
TEST PASSED: No errors detected.
```

**Sample Scene Graph Output:**
```
Detected: person (0.98)
Detected: chair (0.87)
Detected: table (0.76)
```

**Demo Video:**
A short demo video is available in the [llm-camera-tracker/index.html](llm-camera-tracker/index.html) web interface, or you can run the GUI and see live detection overlays.

## Performance Benchmarks & Metrics

| System Type   | Resolution  | Expected FPS | Frame Proc. Time | Detection Time | TTS Latency |
|--------------|-------------|--------------|------------------|---------------|-------------|
| High-end PC  | 1920x1080   | 25-30        | 15-20ms          | 20-30ms       | <200ms      |
| Mid-range PC | 1280x720    | 20-25        | 25-35ms          | 30-40ms       | <250ms      |
| Laptop       | 640x480     | 15-20        | 35-50ms          | 40-60ms       | <300ms      |
| Wearable     | 320x240     | 10-15        | 50-80ms          | 60-100ms      | <350ms      |

- **Detection accuracy**: YOLOv4-tiny (COCO) ~33% mAP (see [YOLOv4-tiny benchmarks](https://github.com/AlexeyAB/darknet)).
- **TTS latency**: Local TTS (pyttsx3) typically <300ms per phrase.
- **Navigation latency**: End-to-end (frame to voice) <500ms on most systems.

## Security & Privacy
- **Video data** is processed in real time and not stored by default. Path recordings save only extracted features, detected object labels, and pose data (not raw video frames).
- **No video or audio is uploaded to the cloud** unless you explicitly enable cloud-based LLM or TTS features.
- **Data storage**: Path and detection data are stored locally in the `demo_paths/` directory as `.pkl` files. You can delete these at any time.
- **Anonymization**: Only object labels and spatial features are saved; no personally identifiable information is retained.
- **User control**: You may clear all saved paths and data via the GUI or by deleting files in `demo_paths/`.

## How to Use
1. **Recording a Path:**
   - Open the GUI and click "Record Path".
   - Enter a path name, destination, and (optionally) a place name for geocoding.
   - Walk the path with your wearable camera. When done, click "Stop Recording".

2. **Navigating:**
   - In the GUI, select a saved path from the list **or** enter a place name to use the geocoding feature.
   - If a path with matching or nearby coordinates exists, EchoWear will guide you along it. Otherwise, you can select a path manually.
   - The system will provide real-time voice guidance and obstacle warnings.
   - If the camera is covered or too dark, EchoWear will say: **"Camera is covered."**

3. **Scene Graphs:**
   - EchoWear uses a scene graph submodule to build a spatial map of detected objects and their relationships, enabling more advanced navigation and context-aware feedback.

4. **Submodule Usage:**
   - The project leverages a dedicated submodule (`llm-camera-tracker`) for camera feed processing, object detection, and scene graph construction. Make sure all submodule dependencies are installed.

## Quick Start

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd EchoWear
```

### 2. Install Dependencies (Locally)
```sh
pip install -r requirements.txt
```

### 3. Run the GUI App
```sh
python gui_app.py
```

### 4. Run Tests
#### Smoke Test
```sh
python smoke_test.py
```
#### Pressure Test
```sh
python pressure_test.py
```

## Docker & Docker Compose

### Build and Run the GUI App
```sh
docker-compose build
docker-compose up
```

### Run Automated Tests in Docker
```sh
docker-compose run --rm echowear-test
```

## Project Structure
- `gui_app.py` — Desktop GUI for navigation and path recording
- `path_manager.py` — Core logic for path management and navigation
- `geocoding_utils.py` — Geocoding helper functions
- `smoke_test.py` — Basic functionality test
- `pressure_test.py` — Stress test for frame processing and TTS
- `Dockerfile`, `docker-compose.yml` — Containerization
- `demo_paths/`, `models/` — Data and model storage
- `llm-camera-tracker/` — Submodule for camera tracking and scene graphing

## Contributing
We welcome contributions! To get started:
- Fork the repository
- Create a new branch for your feature or bugfix
- Submit a pull request with a clear description

**Ideas for contribution:**
- Improve accessibility and user experience
- Add new navigation or feedback features
- Optimize performance for wearable hardware
- Expand test coverage

---

*This project is a prototype and not intended for medical or safety-critical use. For research and educational purposes only.* 
