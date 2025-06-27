import time
import numpy as np
import pyttsx3
from path_manager import PathManager
import sys

print("--- EchoWear Pressure Test ---")

pm = PathManager(storage_dir="demo_paths")

# 1. Stress test process_frame
N_FRAMES = 500
print(f"Processing {N_FRAMES} dummy frames...")
start = time.time()
errors = 0
for i in range(N_FRAMES):
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    try:
        pm.process_frame(frame)
    except Exception as e:
        print(f"Frame {i}: FAIL ({e})")
        errors += 1
end = time.time()
print(f"Frame processing done in {end-start:.2f}s with {errors} errors.")

# 2. Stress test TTS
N_TTS = 100
print(f"Calling TTS {N_TTS} times...")
tts_errors = 0
engine = pyttsx3.init()
start = time.time()
for i in range(N_TTS):
    try:
        engine.say(f"This is TTS test number {i+1}.")
    except Exception as e:
        print(f"TTS {i}: FAIL ({e})")
        tts_errors += 1
engine.runAndWait()
end = time.time()
print(f"TTS done in {end-start:.2f}s with {tts_errors} errors.")

print("--- Pressure Test Complete ---")

if errors > 0 or tts_errors > 0:
    print(f"TEST FAILED: {errors} frame errors, {tts_errors} TTS errors.")
    sys.exit(1)
else:
    print("TEST PASSED: No errors detected.") 