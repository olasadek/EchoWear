# EchoWear: AI Navigation Helper for the Visually Impaired

EchoWear is a prototype AI-powered navigation assistant designed to help visually impaired users navigate indoor and outdoor environments. A protoype for navigation glasses with AI ship. It uses a wearable camera, real-time computer vision, object detection, and voice guidance to provide safe, context-aware navigation instructions.
![b2683635-b3e7-4531-bdb9-9c19ccfc6c88](https://github.com/user-attachments/assets/106ea61b-35cd-4dcf-8884-50ffa628f79b)

**Attribution:**
This is an attribution to doctor Ammar Mohanna and his contributions to the AI society. Thank you for encouraging us in this fast past field.


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
- Uses a modular architecture with a dedicated submodule for camera tracking and scene graphing
- Saves User's favorite path to destination

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
