from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

camera_num = 0
first_frame = 402
last_frame = 440
#person = "Yash"
#FRAMES_FOLDER = f"Video_Camera_Processing/throws/{person}_throw_{camera_num}/png"
#OUTPUT_MP4 = f"Video_Camera_Processing/throws/{person}_throw_{camera_num}/mp4/{person}_throw_{camera_num}.mp4"
FRAMES_FOLDER = "Video_Camera_Processing/throws/throw1/png"
OUTPUT_MP4 = "Video_Camera_Processing/throws/throw1/throw1.mp4"
FPS = 19.0 #CHANGE AS NEEDED
PATTERN = "*.png"
CRF = 23
PIX_FMT = "yuv420p"

ENCODING_BACKEND = "ffmpeg"
COPY_FRAMES_TO_TEMP = True
FORCE_KEYFRAME_EVERY_FRAME = True
MAX_OUTPUT_HEIGHT = None


def list_pngs(folder, pattern, first_idx, last_idx_inclusive):
    names = [n for n in os.listdir(folder) if fnmatch.fnmatch(n, pattern)]
    names = [n for n in names if n.lower().endswith(".png")]
    names.sort()
    n = len(names)
    if n == 0:
        print("Found 0 PNGs")
        return []
    lo = max(0, min(first_idx, n - 1))
    if last_idx_inclusive is None:
        sel = names[lo:]
    else:
        hi = max(lo, min(last_idx_inclusive, n - 1))
        sel = names[lo : hi + 1]
    print(f"Using {len(sel)} of {n} PNGs (sorted indices {lo} through {lo + len(sel) - 1})")
    return [os.path.join(folder, x) for x in sel]


def materialize_frame(src, dst):
    if COPY_FRAMES_TO_TEMP:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def encode_ffmpeg(tmp_dir, nframes, out_mp4):
    seq = os.path.join(tmp_dir, "%06d.png")
    vf = []
    if MAX_OUTPUT_HEIGHT is not None:
        h = int(MAX_OUTPUT_HEIGHT)
        vf = ["-vf", f"scale=-2:{h}"]

    x264 = []
    if FORCE_KEYFRAME_EVERY_FRAME:
        x264 = ["-x264-params", "keyint=1:min-keyint=1:scenecut=0"]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "image2",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
        "-i",
        seq,
        *vf,
        "-frames:v",
        str(nframes),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-profile:v",
        "high",
        "-pix_fmt",
        PIX_FMT,
        "-crf",
        str(CRF),
        "-movflags",
        "+faststart",
        *x264,
        out_mp4,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
    return r.returncode


def encode_opencv(paths, out_mp4):
    import cv2

    first = cv2.imread(paths[0])
    if first is None:
        print(f"OpenCV could not read {paths[0]}", file=sys.stderr)
        return 1
    h0, w0 = first.shape[:2]
    if MAX_OUTPUT_HEIGHT is not None and h0 > MAX_OUTPUT_HEIGHT:
        nh = int(MAX_OUTPUT_HEIGHT)
        nw = int(round(w0 * (nh / float(h0))))
        w_out = max(2, (nw // 2) * 2)
        h_out = max(2, (nh // 2) * 2)
    else:
        w_out, h_out = w0, h0

    out_mp4 = os.path.abspath(out_mp4)
    writer = None
    fourcc_used = None
    for tag in ("avc1", "mp4v", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(out_mp4, fourcc, float(FPS), (w_out, h_out))
        if w.isOpened():
            writer = w
            fourcc_used = tag
            break
    if writer is None:
        print("OpenCV VideoWriter could not open any H.264/MPEG-4 codec for this path.", file=sys.stderr)
        return 1
    print(f"OpenCV encoder: fourcc={fourcc_used}, size={w_out}x{h_out}")

    for p in paths:
        im = cv2.imread(p)
        if im is None:
            print(f"OpenCV could not read {p}", file=sys.stderr)
            writer.release()
            return 1
        if im.shape[1] != w_out or im.shape[0] != h_out:
            im = cv2.resize(im, (w_out, h_out), interpolation=cv2.INTER_AREA)
        writer.write(im)
    writer.release()
    return 0


def ffprobe_summary(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,r_frame_rate,width,height",
        "-of",
        "default=noprint_wrappers=1",
        path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        print("ffprobe:", " | ".join(line.strip() for line in r.stdout.strip().splitlines()))


def main():
    folder = os.path.abspath(FRAMES_FOLDER)
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    paths = list_pngs(folder, PATTERN, first_frame, last_frame)
    if not paths:
        print(f"No PNGs matching {PATTERN!r} in {folder}", file=sys.stderr)
        return 1

    out_mp4 = os.path.abspath(OUTPUT_MP4)
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="png_folder_to_mp4_")
    try:
        for i, p in enumerate(paths):
            dst = os.path.join(tmp_dir, f"{i:06d}.png")
            materialize_frame(os.path.abspath(p), dst)

        if ENCODING_BACKEND == "opencv":
            code = encode_opencv(paths, out_mp4)
        else:
            code = encode_ffmpeg(tmp_dir, len(paths), out_mp4)
        if code != 0:
            return code
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Wrote {out_mp4} ({len(paths)} frames @ {FPS} fps)")
    ffprobe_summary(out_mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
