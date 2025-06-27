import sys
import cv2
import pyttsx3
from path_manager import PathManager

print("--- EchoWear Smoke Test ---")

# 1. Import and instantiate PathManager
try:
    pm = PathManager(storage_dir="demo_paths")
    print("PathManager: OK")
except Exception as e:
    print(f"PathManager: FAIL ({e})")
    sys.exit(1)

# 2. List available paths
try:
    paths = pm.list_paths_with_destinations()
    print(f"Paths found: {len(paths)}")
except Exception as e:
    print(f"List paths: FAIL ({e})")

# 3. Initialize TTS
try:
    engine = pyttsx3.init()
    engine.say("EchoWear smoke test successful.")
    engine.runAndWait()
    print("TTS: OK")
except Exception as e:
    print(f"TTS: FAIL ({e})")

# 4. Try to open the camera and grab a frame
try:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print("Camera: OK (frame captured)")
    else:
        print("Camera: FAIL (no frame)")
    cap.release()
except Exception as e:
    print(f"Camera: FAIL ({e})")

print("--- Smoke Test Complete ---") 