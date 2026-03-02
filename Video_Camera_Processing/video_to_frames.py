import sys
from pathlib import Path

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = _PROJECT_ROOT / "sample_data"


def main():
    if len(sys.argv) >= 2:
        video_path = Path(sys.argv[1])
    else:
        video_path = SAMPLE_DATA / "full_trim_cam2.mp4"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    # Output folder: sample_data/<video_stem>_frames
    out_dir = SAMPLE_DATA / f"{video_path.stem}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"frame_{n:04d}.png"
        cv2.imwrite(str(out_path), frame)
        n += 1
        if n % 100 == 0:
            print(f"  wrote {n} frames...")

    cap.release()
    print(f"Saved {n} frames to {out_dir}")


if __name__ == "__main__":
    main()
