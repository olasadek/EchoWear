# EchoWear: AI-Powered Navigation System with Scene Graph Intelligence

EchoWear is an advanced AI-powered navigation assistant designed to help visually impaired users navigate indoor and outdoor environments with intelligent scene understanding. This prototype demonstrates the future of wearable AI navigation glasses with real-time computer vision, object detection, scene graph construction, and voice guidance.

![image](https://github.com/user-attachments/assets/0454f07d-f029-4328-9a3b-eb9a5e811715)





## Updates (Latest Release)

### ✨ Scene Graph Intelligence
- **Hierarchical Scene Understanding**: Advanced scene graph construction using hierarchical graph builders
- **Real-time Object Relationship Mapping**: Detects and maps spatial relationships between objects
- **Intelligent Obstacle Classification**: Prioritizes obstacles (people > large objects > small objects)
- **Context-Aware Navigation**: Uses scene context for more intelligent navigation decisions
- **Natural Language Scene Descriptions**: Converts visual scenes to natural language for better understanding

### 🐳 Enhanced Docker Support
- **Multi-Service Architecture**: Separate containers for GUI, tests, demos, and scene graph server
- **Comprehensive Dependencies**: All required packages pre-installed and configured
- **Health Checks**: Built-in health monitoring for all services
- **Easy Management**: Simple script (`docker-run.sh`) for managing all services
- **Optimized Builds**: `.dockerignore` for faster, more efficient builds

### 🎯 New Features
- **GPS Navigation with Obstacle Detection**: Real-time GPS navigation with camera-based obstacle detection
<img width="655" alt="gps" src="https://github.com/user-attachments/assets/761ddc08-8154-45a0-a25e-1c37233ad216" />

- **Enhanced TTS Voice Guidance**: Comprehensive voice feedback with obstacle warnings
- **Multi-Path Management**: Create and manage multiple navigation paths with destinations
- **Scene Graph Visualization**: Visual representation of detected objects and relationships
- **Advanced Error Handling**: Robust error handling and fallback mechanisms

## 🧠 Scene Graph Technology

### Core Components
- **HierarchicalGraphBuilder**: Builds multi-level scene representations
- **SceneGraphBuilder**: Creates scene graphs from object detections
- **GraphStore**: Manages and persists scene graph data
- **GraphVisualizer**: Visualizes scene graphs and relationships

### Scene Understanding Features
```python
# Example scene graph output
Scene: "There is a person in the scene."
Obstacle: "Person detected ahead - be careful"
Scene Graph: {
    "entities": ["person"],
    "relationships": ["person -> ahead"],
    "importance": 0.95,
    "timestamp": 1234567890
}
```

### Intelligent Obstacle Detection
1. **Person Detection** (Highest Priority): "Person detected ahead - be careful"
2. **Large Object Detection**: "Large object detected ahead"
3. **Small Object Detection**: "Small object detected ahead"
4. **Motion Detection**: "Moving object detected ahead"
5. **Dark Object Detection**: "Dark object detected ahead"

## 🏗️ Architecture Overview

### Core Modules
- **PathManager**: Visual odometry-based path recording and navigation
- **LLMNavigator**: AI-powered navigation with TTS and fallback logic
- **SceneGraph**: Advanced scene understanding and object relationship mapping
- **CameraProcessor**: Real-time camera feed processing with object detection
- **GUI**: Modern Tkinter-based user interface with scene graph integration

### Data Flow
```
Camera Feed → Object Detection → Scene Graph → Navigation Logic → TTS Voice Guidance
     ↓              ↓                ↓              ↓                ↓
Visual Odometry → Keypoint Matching → Spatial Mapping → Path Guidance → User Feedback
```

## 📦 Installation & Setup

### Quick Start with Docker (Recommended)

1. **Clone and Setup**
```bash
git clone https://github.com/olasadek/EchoWear.git
cd EchoWear
chmod +x docker-run.sh
```

2. **Build and Run**
```bash
# Build the Docker image
./docker-run.sh build

# Run the GUI
./docker-run.sh gui

# Run tests
./docker-run.sh test

# Run camera test
./docker-run.sh camera-test

# Run enhanced demo
./docker-run.sh demo

# Run scene graph server
./docker-run.sh server
```

### Manual Installation

1. **Install System Dependencies (Linux)**
```bash
sudo apt-get update && sudo apt-get install -y \
    python3-tk libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 \
    libxext6 alsa-utils espeak ffmpeg pulseaudio libespeak1 \
    libgomp1 libgcc-s1 libstdc++6 libc6 libblas3 liblapack3 \
    libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev \
    libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-dev \
    build-essential pkg-config libssl-dev libffi-dev
```

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Run the Application**
```bash
python gui_app.py
```

## 🎮 Usage Guide

### 1. Path Recording with Scene Graph
```bash
# Run the enhanced demo for path recording
./docker-run.sh demo
# or
python enhanced_path_manager_demo.py
```

**Features:**
- Real-time scene descriptions during recording
- Object detection and classification
- Destination labeling with geocoding
- Scene graph data collection

### 2. GPS Navigation with Obstacle Detection
```bash
# Use the GUI's "GPS Navigation" button
# or run demo 6 directly
python enhanced_path_manager_demo.py
```

**Features:**
- Current location detection
- Nearby place search
- Real-time obstacle detection
- Voice guidance with cardinal directions
- Distance and bearing calculations

### 3. Scene Graph Server
```bash
# Start the scene graph server
./docker-run.sh server
```

**Access the web interface at:** `http://localhost:8000`

### 4. Camera Testing
```bash
# Test camera functionality
./docker-run.sh camera-test
# or
python test_camera.py
```

## 🔧 Docker Services

### Available Services
- **echowear-gui**: Main GUI application with scene graph integration
- **echowear-test**: Automated testing suite
- **echowear-demo**: Enhanced path manager demo
- **echowear-camera-test**: Camera functionality testing
- **scene-graph-server**: Web-based scene graph visualization server

### Docker Management Commands
```bash
# View all available commands
./docker-run.sh help

# Check service status
./docker-run.sh status

# View logs
./docker-run.sh logs [service-name]

# Clean up resources
./docker-run.sh cleanup
```

## 📊 Performance & Benchmarks

### Scene Graph Performance
- **Object Detection**: YOLOv4-tiny ~33% mAP
- **Scene Graph Construction**: <100ms per frame
- **Relationship Mapping**: Real-time spatial analysis
- **TTS Latency**: <300ms per phrase

### System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| GPU | Integrated | Dedicated GPU |
| Camera | 640x480 | 1280x720+ |
| Storage | 2GB | 5GB+ |

### Performance Metrics
- **Frame Processing**: 15-50ms depending on resolution
- **Object Detection**: 20-60ms per frame
- **Scene Graph Update**: <100ms per scene
- **Navigation Latency**: <500ms end-to-end
- **TTS Response**: <300ms per instruction

## 🧪 Testing & Validation

### Automated Tests
```bash
# Run all tests
./docker-run.sh test

# Individual test files
python smoke_test.py      # Basic functionality
python pressure_test.py   # Stress testing
python test_camera.py     # Camera testing
```

### Test Coverage
- **Smoke Test**: Core functionality validation
- **Pressure Test**: High-load performance testing
- **Camera Test**: Camera hardware validation
- **Scene Graph Test**: Scene understanding validation
- **TTS Test**: Voice guidance validation

## 🔒 Security & Privacy

### Data Protection
- **Local Processing**: All video processing happens locally
- **No Cloud Upload**: No video or audio data is uploaded
- **Feature Extraction**: Only spatial features and object labels are stored
- **User Control**: Complete control over saved data

### Privacy Features
- **Anonymized Data**: No personally identifiable information
- **Local Storage**: All data stored locally in `demo_paths/`
- **Easy Deletion**: Clear all data via GUI or file deletion
- **No Tracking**: No analytics or tracking mechanisms

## 🛠️ Development & Contributing

### Development Setup
```bash
# Clone with submodules
git clone --recursive <your-repo-url>
cd EchoWear

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

### Project Structure
```
EchoWear/
├── gui_app.py                    # Main GUI application
├── path_manager.py               # Core path management
├── llm_navigator.py              # AI navigation logic
├── enhanced_path_manager_demo.py # Enhanced demo with scene graphs
├── geocoding_utils.py            # GPS and geocoding utilities
├── test_camera.py                # Camera testing utility
├── smoke_test.py                 # Basic functionality tests
├── pressure_test.py              # Performance stress tests
├── llm-camera-tracker/           # Scene graph submodule
│   ├── scene_graph/              # Scene graph components
│   ├── camera_processor/         # Camera processing pipeline
│   └── scene_graph_server.py     # Web server for visualization
├── models/                       # ML model storage
├── demo_paths/                   # Recorded path data
├── output/                       # Generated outputs
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Multi-service orchestration
├── docker-run.sh                 # Docker management script
└── requirements.txt              # Python dependencies
```

# Run tests
pytest

# Run with debugging
python -m pdb gui_app.py
```

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

**Areas for Contribution:**
- Improved object detection models
- Enhanced scene graph algorithms 
- Better TTS and voice guidance
- Performance optimizations
- Accessibility improvements
- Additional navigation features

## 🚨 Troubleshooting

### Common Issues

**Camera Not Working:**
```bash
# Test camera functionality
./docker-run.sh camera-test

# Check camera permissions
ls -la /dev/video*

# Reset camera in GUI
# Use the "Reset Camera" button
```

**Scene Graph Errors:**
```bash
# Install spaCy model
python -m spacy download en_core_web_sm

# Check spaCy installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')"
```

**Docker Issues:**
```bash
# Clean up Docker resources
./docker-run.sh cleanup

# Rebuild image
./docker-run.sh build

# Check Docker status
docker system info
```

**TTS Not Working:**
```bash
# Install system audio dependencies
sudo apt-get install alsa-utils espeak ffmpeg

# Test TTS
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*⚠️ Disclaimer: This project is a prototype for research and educational purposes. It is not intended for medical or safety-critical use.*
*In honor of DR. Ammar Mohanna and his invaluable contributions to the AI community.*
