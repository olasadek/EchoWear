import pickle
import sys
import numpy as np

def inspect_path(path_file):
    with open(path_file, 'rb') as f:
        data = pickle.load(f)
    frames = data['frames']
    poses = data['poses']
    print(f"Path file: {path_file}")
    print(f"Number of frames: {len(frames)}")
    print(f"Number of poses: {len(poses)}")
    if len(frames) > 0:
        print("\nSample frame[0] keypoints (first 5):")
        kps = frames[0]['keypoints']
        if kps is not None:
            for i, kp in enumerate(kps[:5]):
                print(f"  Keypoint {i}: {kp}")
        else:
            print("  No keypoints in first frame.")
    if len(poses) > 0:
        print("\nSample pose[0]:")
        print(np.array(poses[0]))
    if len(poses) > 1:
        print("\nSample pose[1]:")
        print(np.array(poses[1]))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_path_file.py <path_file>")
    else:
        inspect_path(sys.argv[1]) 