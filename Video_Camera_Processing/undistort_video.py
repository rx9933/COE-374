import sys
from pathlib import Path

import cv2
import numpy as np

# Path to K (project root = parent of Video_Camera_Processing)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
K_PATH = _PROJECT_ROOT / "Video_Camera_Processing" / "K_list.npy"
CAMERA_INDEX = 2
# Distortion coefficients (k1, k2, p1, p2, k3)
DIST = np.array([-0.30, 0.10, 0.0, 0.0, 0.0], dtype=np.float64)


def main():
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    else:
        video_path = f"sample_data/full_trim_cam{CAMERA_INDEX}.mp4"
        out_path = None

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    if out_path is None:
        out_path = video_path.parent / f"{video_path.stem}_undistorted{video_path.suffix}"
    out_path = Path(out_path)

    if not K_PATH.exists():
        print(f"K not found: {K_PATH}. Run calibration or place K.npy in Video_Camera_Processing.")
        sys.exit(1)

    K = np.load(K_PATH)
    K = K[CAMERA_INDEX - 1] # use the second camera
    if K.shape != (3, 3):
        print(f"K must be 3x3, got shape {K.shape}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Precompute undistortion map (once)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, DIST, None, K, (w, h), cv2.CV_32FC1
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"Undistorting: {video_path} -> {out_path} ({n_frames} frames, {w}x{h} @ {fps} fps)")

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        writer.write(undistorted)
        t += 1
        if t % 100 == 0 or t == n_frames:
            print(f"  Frame {t}/{n_frames}")

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
