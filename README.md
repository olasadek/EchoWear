# EchoWear: AI Navigation Helper for the Visually Impaired

EchoWear is a prototype AI-powered navigation assistant designed to help visually impaired users navigate indoor and outdoor environments. It uses a wearable camera, real-time computer vision, object detection, and voice guidance to provide safe, context-aware navigation instructions.

## Features
- Real-time camera feed processing
- Visual path recording and navigation
- Object detection and obstacle warnings
- Voice guidance (TTS)
- Geocoding for destination selection
- Desktop GUI (Tkinter)
- Dockerized for easy deployment
- Automated smoke and pressure tests

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