import sys
import subprocess
import re


def get_video_info(video_path):
    """Return (n_frames, width, height, fps) using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr or result.stdout}")

    out = result.stdout
    width = int(re.search(r"width=(\d+)", out).group(1))
    height = int(re.search(r"height=(\d+)", out).group(1))
    rate_match = re.search(r"r_frame_rate=(\d+)/(\d+)", out)
    if rate_match:
        num, den = int(rate_match.group(1)), int(rate_match.group(2))
        fps = num / den if den else 0.0
    else:
        fps = 0.0

    n_frames = None
    nb_match = re.search(r"nb_frames=(\d+)", out)
    if nb_match:
        n_frames = int(nb_match.group(1))
    if n_frames is None:
        dur_match = re.search(r"duration=([\d.]+)", out)
        if dur_match and fps > 0:
            duration = float(dur_match.group(1))
            n_frames = int(round(duration * fps))

    return n_frames, width, height, fps


if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = "sample_data/full_trim_cam1.mp4"

try:
    n_frames, width, height, fps = get_video_info(video_path)
except FileNotFoundError:
    print("ffprobe not found. Install ffmpeg (e.g. brew install ffmpeg).")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print(f"Video: {video_path}")
print(f"  Frames:  {n_frames if n_frames is not None else 'unknown'}")
print(f"  Resolution: {width} x {height}")
print(f"  FPS: {fps}")
