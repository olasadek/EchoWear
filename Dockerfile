FROM python:3.10-slim

# Install system dependencies for OpenCV, Tkinter, TTS, and sound
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
        && rm -rf /var/lib/apt/lists/*

# Set environment variables for Tkinter and sound
ENV DISPLAY=:0
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app
WORKDIR /app

# Default command
CMD ["python", "gui_app.py"] 