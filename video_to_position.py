"""
Video to 3D position estimation pipeline.

Runs CVScripts/video_detection and MathScripts/Physics_Triangulation_No_Camera_Conditioning.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "MathScripts"))

from CVScripts.better_video_detection import (
    extract_trajectory_from_video,
    PROCESS_WIDTH,
    PROCESS_HEIGHT,
)
from MathScripts.Physics_Triangulation_No_Camera_Conditioning import optimize_trajectory

PROCESS_SIZE = (PROCESS_WIDTH, PROCESS_HEIGHT)


def run_pipeline(video_paths, P_list, dt, g, orig_sizes, pixel_sigma=1.0, physics_sigma=0.01, max_frames=None, omega_phys=1.0):
    n_cameras = len(video_paths)
    if n_cameras < 2:
        raise ValueError("At least 2 cameras (videos) required.")
    if len(P_list) != n_cameras:
        raise ValueError(f"P_list has {len(P_list)} cameras but {n_cameras} videos provided.")
    if len(orig_sizes) != n_cameras:
        raise ValueError(f"orig_sizes has {len(orig_sizes)} entries but {n_cameras} cameras.")

    positions_per_camera = []
    detected_per_camera = []
    for vp in video_paths:
        pos_list, det_list, _, _ = extract_trajectory_from_video(vp, max_frames=max_frames)
        positions_per_camera.append(pos_list)
        detected_per_camera.append(det_list)

    n_frames_per_cam = [len(pos_list) for pos_list in positions_per_camera]
    n_common = min(n_frames_per_cam)
    if any(n != n_common for n in n_frames_per_cam):
        positions_per_camera = [pos_list[-n_common:] for pos_list in positions_per_camera]
        detected_per_camera = [det_list[-n_common:] for det_list in detected_per_camera]
        print(f"Trimmed to {n_common} frames (dropped first frames from longer videos).")
    n_frames_raw = n_common

    valid_t = [
        t for t in range(n_frames_raw)
        if all(positions_per_camera[i][t] is not None for i in range(n_cameras))
    ]
    positions_all_frames = [list(positions_per_camera[i]) for i in range(n_cameras)]
    detected_all_frames = [list(detected_per_camera[i]) for i in range(n_cameras)]
    positions_per_camera = [
        [pos_list[t] for t in valid_t]
        for pos_list in positions_per_camera
    ]
    n_frames = len(valid_t)

    if n_frames < 3:
        raise ValueError(
            f"Need at least 3 frames with detections in all cameras. "
            f"After removing frames with missing detections, got {n_frames} frames."
        )

    if n_frames < n_frames_raw:
        print(f"Dropped {n_frames_raw - n_frames} frames with missing detections; using {n_frames} frames.")

    pixels_for_draw = [[positions_per_camera[i][t] for t in range(n_frames)] for i in range(n_cameras)]
    pixels = []

    print("This is pixels_for_draw: \n", pixels_for_draw)
    for i in range(n_cameras):
        w_orig, h_orig = orig_sizes[i]
        scaled = []
        for t in range(n_frames):
            cx, cy = pixels_for_draw[i][t][0], pixels_for_draw[i][t][1]
            scaled.append(np.array([
                cx * w_orig / PROCESS_WIDTH,
                cy * h_orig / float(PROCESS_HEIGHT),
            ], dtype=np.float64))
        pixels.append(scaled)

    #print("Length of pixels: \n", len(pixels))
    X_opt, cov, _ = optimize_trajectory(
        P_list, pixels, dt=dt, g=np.asarray(g, dtype=np.float64), drag=0.0,
        pixel_sigma=pixel_sigma, physics_sigma=physics_sigma, omega_phys=omega_phys,
    )
    start_per_cam = [n_frames_per_cam[i] - n_common for i in range(n_cameras)]
    frame_indices = [
        [start_per_cam[i] + valid_t[t] for t in range(n_frames)]
        for i in range(n_cameras)
    ]
    frame_indices_all = [[start_per_cam[i] + t for t in range(n_frames_raw)] for i in range(n_cameras)]
    return X_opt, cov, frame_indices, pixels_for_draw, frame_indices_all, positions_all_frames, detected_all_frames, pixels


def plot_3d_trajectory(X_opt, cov=None, out_path="trajectory_3d.png"):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Install matplotlib to plot: pip install matplotlib")
        return
    n = len(X_opt)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    mask = X_opt[:, 0] > -10000
    X_plot = X_opt[mask]
    frame_index = np.arange(n)[mask]
    sc = ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=frame_index, cmap="viridis", s=200, edgecolors="none")
    ax.plot(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], "k-", alpha=0.25, linewidth=1.5)

    #if cov is not None and np.all(np.isfinite(cov)):
    #    max_std = np.max(np.sqrt(np.diag(cov)))
    #    scale = 0.3 / max_std if max_std > 1.0 else 1.0
    #    for t in range(0, n, max(1, n // 15)):
    #        cov_t = cov[3 * t : 3 * t + 3, 3 * t : 3 * t + 3]
    #        if np.any(np.isnan(cov_t)) or np.any(np.linalg.eigvalsh(cov_t) <= 0):
    #            continue
    #        eigs, Q = np.linalg.eigh(cov_t)
    #        eigs = np.maximum(eigs, 1e-12)
    #        radii = scale * np.sqrt(eigs)
    #        u = np.linspace(0, 2 * np.pi, 20)
    #        v = np.linspace(0, np.pi, 20)
    #        x = np.outer(np.cos(u), np.sin(v))
    #        y = np.outer(np.sin(u), np.sin(v))
    #        z = np.outer(np.ones_like(u), np.cos(v))
    #        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) @ (Q * radii).T + X_opt[t]
    #        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="gray", alpha=0.15, s=1)
    ax.set_xlabel("x (m)", fontsize=20)
    ax.set_ylabel("y (m)", fontsize=20)
    ax.set_zlabel("z (m)", fontsize=20)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Frame", fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.show()

def plot_2d_pixels(pixels, dimensions, out_path="trajectory_2d.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to plot: pip install matplotlib")
        return
    pixels = np.asarray(pixels)
    print("This is pixels: \n", pixels)
    print("u range:", pixels[:, 0].min(), pixels[:, 0].max())
    print("v range:", pixels[:, 1].min(), pixels[:, 1].max())
    print("frame size:", dimensions)
    n = len(pixels)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sc = ax.scatter(pixels[:, 0], pixels[:, 1], c=[i for i in range(n)], cmap="viridis", s=60, edgecolors="none")
    ax.plot(pixels[:, 0], pixels[:, 1], "k-", alpha=0.25, linewidth=1.5)
    ax.set_xlabel("u (pixels)", fontsize=20)
    ax.set_ylabel("v (pixels)", fontsize=20)
    #make lines at dimensions
    ax.axvline(x=dimensions[0], color='r', linestyle='--')
    ax.axhline(y=dimensions[1], color='r', linestyle='--')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.axhline(y=0, color='r', linestyle='--')
    #make the size of the plot the same as the frame size
    ax.set_xlim(0, dimensions[0])
    ax.set_ylim(0, dimensions[1])
    ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Frame", fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.show()

def save_frames(video_paths, frame_indices, out_dir, pixels=None, detected=None, process_size=PROCESS_SIZE, box_half_size=30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cameras = len(video_paths)
    n_frames = len(frame_indices[0])
    pw, ph = process_size

    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if not all(cap.isOpened() for cap in caps):
        for cap in caps:
            cap.release()
        raise RuntimeError("Could not open one or more videos.")

    for t in range(n_frames):
        frames = []
        for i in range(n_cameras):
            cap = caps[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i][t])
            ret, frame = cap.read()
            if not ret:
                for cap in caps:
                    cap.release()
                raise RuntimeError(f"Could not read frame {frame_indices[i][t]} from video {i}.")
            h, w = frame.shape[:2]
            font_scale = max(0.6, min(2.5, w / 600.0))
            thickness = max(2, int(round(w / 500)))
            has_position = pixels is not None and i < len(pixels) and t < len(pixels[i]) and pixels[i][t] is not None
            is_real_detection = (
                detected is not None and i < len(detected) and t < len(detected[i]) and detected[i][t]
            )
            if has_position and (is_real_detection if detected is not None else True):
                cx_proc, cy_proc = float(pixels[i][t][0]), float(pixels[i][t][1])
                cx = cx_proc * w / pw
                cy = cy_proc * h / ph
                box_half_w = max(15, int(round(box_half_size * w / pw)))
                box_half_h = max(15, int(round(box_half_size * h / ph)))
                x1 = int(round(cx)) - box_half_w
                y1 = int(round(cy)) - box_half_h
                x2 = int(round(cx)) + box_half_w
                y2 = int(round(cy)) + box_half_h
                x1 = max(0, min(w - 2, x1))
                y1 = max(0, min(h - 2, y1))
                x2 = max(x1 + 4, min(w, x2))
                y2 = max(y1 + 4, min(h, y2))
                line_thickness = max(2, thickness)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
                status_text = "Detected"
                status_color = (0, 255, 0)
            elif has_position and not (is_real_detection if detected is not None else True):
                cx_proc, cy_proc = float(pixels[i][t][0]), float(pixels[i][t][1])
                cx = cx_proc * w / pw
                cy = cy_proc * h / ph
                pt = (int(round(cx)), int(round(cy)))
                marker_size = max(24, int(round(40 * w / pw)))
                cv2.drawMarker(frame, pt, (0, 165, 255), cv2.MARKER_CROSS, marker_size, thickness)
                status_text = "Predicted (no detection)"
                status_color = (0, 165, 255)
            else:
                status_text = "No bounding box found"
                status_color = (0, 0, 255)
            cv2.putText(frame, f"Cam {i}: {status_text}", (20, 40 + int(40 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
            if status_text == "No bounding box found":
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = (w - tw) // 2
                ty = (h + th) // 2
                cv2.putText(frame, status_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
            frames.append(frame)

        max_h = max(f.shape[0] for f in frames)
        max_w = max(f.shape[1] for f in frames)
        padded = []
        for frame in frames:
            h, w = frame.shape[:2]
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[:] = (0, 0, 0)
            y0 = (max_h - h) // 2
            x0 = (max_w - w) // 2
            canvas[y0 : y0 + h, x0 : x0 + w] = frame
            padded.append(canvas)
        side_by_side = np.hstack(padded)
        out_path = out_dir / f"frame_{t:04d}.png"
        cv2.imwrite(str(out_path), side_by_side)

    for cap in caps:
        cap.release()
    print(f"Saved {n_frames} side-by-side frames to {out_dir}")


def main():
    video_paths = [
        "sample_data/full_trim_cam1.mp4",
        "sample_data/full_trim_cam2.mp4",
    ]
    P_list_path = "Video_Camera_Processing/P_list.npy"
    dt = 1.0 / 60.0
    g = [0.0, 0.0, -9.81]
    pixel_sigma = 1.0
    physics_sigma = 0.1
    omega_phys = 0.0
    max_frames = None
    out_path = "trajectory_3d.png"
    side_by_side_dir = _PROJECT_ROOT / "sample_data" / "trajectory_side_by_side"
    video_paths = [Path(p) for p in video_paths]
    for p in video_paths:
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")

    P_list = np.load(P_list_path)
    if P_list.ndim != 3 or P_list.shape[1] != 3 or P_list.shape[2] != 4:
        raise ValueError("P_list must have shape (n_cameras, 3, 4)")
    P_list = [P_list[i] for i in range(len(P_list))]

    orig_sizes = [(1920, 1440), (3840, 2160)]

    X_opt, cov, frame_indices, pixels, frame_indices_all, pixels_all_frames, detected_all_frames, pixels_for_camera = run_pipeline(
        [str(p) for p in video_paths],
        P_list=P_list,
        dt=dt,
        g=g,
        orig_sizes=orig_sizes,
        pixel_sigma=pixel_sigma,
        physics_sigma=physics_sigma,
        max_frames=max_frames,
        omega_phys=omega_phys,
    )

    print(f"Estimated 3D trajectory: {len(X_opt)} frames, dt={dt} s")
    print(f"  x range: [{X_opt[:, 0].min():.3f}, {X_opt[:, 0].max():.3f}] m")
    print(f"  y range: [{X_opt[:, 1].min():.3f}, {X_opt[:, 1].max():.3f}] m")
    print(f"  z range: [{X_opt[:, 2].min():.3f}, {X_opt[:, 2].max():.3f}] m")

    for i in range(len(X_opt)):
        print("--------------------------------")
        print(f"Frame {i}")
        print(f"Pixel: {pixels_for_camera[0][i]}, {pixels_for_camera[1][i]}")
        print(f"Position: {X_opt[i]}")
        print("--------------------------------")


    plot_3d_trajectory(X_opt, cov=cov, out_path=out_path)

    for i in range(len(pixels)):
        plot_2d_pixels(pixels_for_camera[i], dimensions=orig_sizes[i], out_path=f"trajectory_2d_{i}.png")

    

    save_frames(video_paths, frame_indices_all, side_by_side_dir, pixels=pixels_all_frames, detected=detected_all_frames)


if __name__ == "__main__":
    main()
