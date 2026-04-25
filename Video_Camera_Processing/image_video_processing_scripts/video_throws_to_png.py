"""
Extract every frame from throw videos to PNG sequences.

Default layout (edit paths at bottom):
  Video_Camera_Processing/throws/filtered_cam_0/frame_000000.png
  Video_Camera_Processing/throws/filtered_cam_1/frame_000000.png

Run from repo root:
  python3 Video_Camera_Processing/image_video_processing_scripts/video_throws_to_png.py
"""

from __future__ import annotations

import os
import sys

import cv2

# Default: two camera videos under throws → sibling folders filtered_cam_0 / filtered_cam_1
THROWS_DIR = "Video_Camera_Processing/throws"
VIDEO_CAM0 = os.path.join(THROWS_DIR, "roi_priority.mp4")
VIDEO_CAM1 = os.path.join(THROWS_DIR, "roi_priority (1).mp4")
OUT_CAM0_DIR = os.path.join(THROWS_DIR, "delete_later_0")
OUT_CAM1_DIR = os.path.join(THROWS_DIR, "delete_later_1")

FRAME_NAME_FMT = "frame_%06d.png"
START_INDEX = 0
MAX_FRAMES = None


def extract_video_to_png_folder(
    video_path: str,
    out_dir: str,
    frame_name_fmt: str = FRAME_NAME_FMT,
    start_index: int = START_INDEX,
    max_frames: int | None = MAX_FRAMES,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_written = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and n_written >= max_frames:
            break
        out_name = frame_name_fmt % (start_index + n_written)
        out_path = os.path.join(out_dir, out_name)
        if not cv2.imwrite(out_path, frame):
            cap.release()
            raise RuntimeError(f"Failed to write {out_path}")
        n_written += 1
        frame_idx += 1

    cap.release()
    return n_written


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) >= 4:
        v0, v1, d0, d1 = argv[0], argv[1], argv[2], argv[3]
    elif len(argv) == 0:
        v0, v1, d0, d1 = VIDEO_CAM0, VIDEO_CAM1, OUT_CAM0_DIR, OUT_CAM1_DIR
    else:
        print(
            "Usage: video_throws_to_png.py [video0 video1 out_dir_cam0 out_dir_cam1]\n"
            "Defaults use THROWS_DIR and filtered_cam_0 / filtered_cam_1 (edit script)."
        )
        sys.exit(1)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(repo_root, p)

    v0, v1 = _resolve(v0), _resolve(v1)
    d0, d1 = _resolve(d0), _resolve(d1)

    for v in (v0, v1):
        if not os.path.isfile(v):
            raise FileNotFoundError(v)

    n0 = extract_video_to_png_folder(v0, d0)
    n1 = extract_video_to_png_folder(v1, d1)
    print(f"Wrote {n0} frames to {d0}")
    print(f"Wrote {n1} frames to {d1}")


if __name__ == "__main__":
    main()
