import cv2
import time
from path_manager import PathManager

def main():
    path_name = input("Enter path name to record: ")
    
    # Initialize PathManager with object detection enabled
    path_manager = PathManager(storage_dir="paths", enable_object_detection=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    
    print("Recording path with object detection. Press 'q' to stop.")
    path_manager.start_recording_path(path_name)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Process frame (includes object detection)
        path_manager.process_frame(frame)
        
        # Display frame with detection info
        display_frame = frame.copy()
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add recording indicator
        cv2.putText(display_frame, "RECORDING", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Recording Path - Press 'q' to stop", display_frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    path_manager.stop_recording_path()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Recorded {frame_count} frames for path '{path_name}'.")
    print("Use 'python inspect_path_with_objects.py' to see detected objects.")

if __name__ == "__main__":
    main() 