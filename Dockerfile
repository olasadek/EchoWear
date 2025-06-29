FROM python:3.10-slim

# Install system dependencies for OpenCV, Tkinter, TTS, sound, and additional libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-tk \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        alsa-utils \
        espeak \
        ffmpeg \
        pulseaudio \
        libespeak1 \
        libgomp1 \
        libgcc-s1 \
        libstdc++6 \
        libc6 \
        libblas3 \
        liblapack3 \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf5-103 \
        libqtgui4 \
        libqtwebkit4 \
        libqt4-test \
        python3-dev \
        build-essential \
        pkg-config \
        libssl-dev \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Set environment variables for Tkinter, sound, and OpenCV
ENV DISPLAY=:0
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be missing
RUN pip install --no-cache-dir \
    geocoder \
    geopy \
    matplotlib \
    pyttsx3 \
    ctransformers \
    sentence-transformers \
    && rm -rf ~/.cache/pip

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the app
COPY . .

# Create necessary directories
RUN mkdir -p demo_paths models output

# Set permissions
RUN chmod +x *.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2; print('OpenCV available')" || exit 1

# Default command
CMD ["python", "gui_app.py"] 