"""
Video to 3D position — manual pixel coordinates in each video's native resolution (no downscale).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import time

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "MathScripts"))

can_save_frames = True
can_plot_3d = True
from MathScripts.Physics_Triangulation_No_Camera_Conditioning import optimize_trajectory

cam_0_pos = [
    2912, 1410, 2896, 1374, 2873, 1341, 2858, 1311, 2836, 1289, 2816, 1266, 2798, 1253, 2777, 1231,
    2758, 1223, 2733, 1218, 2713, 1216, 2690, 1212, 2666, 1216, 2643, 1223, 2616, 1232, 2593, 1252,
    2571, 1268, 2540, 1298, 2511, 1336, 2481, 1370, 2453, 1406, 2426, 1464, 2394, 1511, 2366, 1567,
    2331, 1641, 2299, 1705, 2264, 1793, 2235, 1880, 2191, 1966, 2151, 2074, #2117, 2190
]

cam_1_pos = [
    1732, 1425, 1755, 1395, 1779, 1359, 1801, 1331, 1825, 1296, 1852, 1266, 1875, 1245, 1901, 1233,
    1927, 1214, 1953, 1204, 1982, 1199, 2014, 1196, 2044, 1196, 2071, 1206, 2110, 1216, 2139, 1226,
    2174, 1253, 2210, 1277, 2241, 1307, 2280, 1342, 2316, 1385, 2360, 1440, 2400, 1494, 2444, 1560,
    2490, 1628, 2536, 1710, 2581, 1802, 2634, 1893, 2687, 1997, 2738, 2117, #2797, 2242
]


def read_video_frame_size(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid frame size from {path}: {w}x{h}")
    return w, h


def _point_from_xy(x, y):
    if x is None or y is None:
        return None
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf) or not np.isfinite(yf):
        return None
    return np.array([xf, yf], dtype=np.float64)


def _xy_pairs_from_cam_pos(cam_pos):
    if not cam_pos:
        return []
    first = cam_pos[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and not isinstance(first, (int, float, np.floating, np.integer)):
        pairs = []
        for p in cam_pos:
            pairs.append((p[0], p[1]))
        return pairs
    flat = list(cam_pos)
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def run_pipeline_manual(cam_0_pos, cam_1_pos, P_list, orig_sizes, dt, g, pixel_sigma=1.0, physics_sigma=0.01, omega_phys=1.0):
    n_cameras = 2
    if len(P_list) != n_cameras or len(orig_sizes) != n_cameras:
        raise ValueError("Expected 2 cameras: P_list and orig_sizes length 2.")

    pairs0 = _xy_pairs_from_cam_pos(cam_0_pos)
    pairs1 = _xy_pairs_from_cam_pos(cam_1_pos)
    n0, n1 = len(pairs0), len(pairs1)
    if n0 != n1:
        raise ValueError(f"cam_0_pos has {n0} (x,y) frames, cam_1_pos has {n1}; counts must match.")

    n_frames_raw = n0
    start_per_cam = [0, 0]
    all_pairs = [pairs0, pairs1]

    positions_full = []
    for i in range(n_cameras):
        row = []
        for t in range(n_frames_raw):
            x, y = all_pairs[i][t]
            row.append(_point_from_xy(x, y))
        positions_full.append(row)

    valid_t = [
        t for t in range(n_frames_raw)
        if all(positions_full[i][t] is not None for i in range(n_cameras))
    ]

    positions_all_frames = []
    detected_all_frames = []
    for i in range(n_cameras):
        full_row = []
        det_row = []
        for t in range(n_frames_raw):
            p = positions_full[i][t]
            if p is None:
                full_row.append(None)
                det_row.append(False)
            else:
                full_row.append(p.copy())
                det_row.append(True)
        positions_all_frames.append(full_row)
        detected_all_frames.append(det_row)

    positions_kept = [[positions_full[i][t] for t in valid_t] for i in range(n_cameras)]
    n_frames = len(valid_t)

    if n_frames < 3:
        raise ValueError(f"Need at least 3 frames with points on both cameras; got {n_frames} after filtering.")

    if n_frames < n_frames_raw:
        valid_set = set(valid_t)
        dropped_common_t = [t for t in range(n_frames_raw) if t not in valid_set]
        print(f"Dropped {n_frames_raw - n_frames} frames with missing manual points; using {n_frames} frames.")
        print(f"  Aligned-window indices dropped (0 .. {n_frames_raw - 1}): {dropped_common_t}")
        print("  Original 0-based video frame index per camera for those slots:")
        for i in range(n_cameras):
            print(f"    camera {i}: {[start_per_cam[i] + t for t in dropped_common_t]}")
        for t in dropped_common_t:
            missing = [j for j in range(n_cameras) if positions_full[j][t] is None]
            print(f"    t={t} -> missing cameras {missing}")

    pixels_for_draw = [[positions_kept[i][k].copy() for k in range(n_frames)] for i in range(n_cameras)]
    pixels = [[positions_kept[i][k].copy() for k in range(n_frames)] for i in range(n_cameras)]

    X_opt, cov, _drag_opt, _ls_result = optimize_trajectory(
        P_list,
        pixels,
        dt=dt,
        g=np.asarray(g, dtype=np.float64),
        drag=0.0,
        pixel_sigma=pixel_sigma,
        physics_sigma=physics_sigma,
        omega_phys=omega_phys,
    )
    frame_indices = [
        [start_per_cam[i] + valid_t[t] for t in range(n_frames)] for i in range(n_cameras)
    ]
    frame_indices_all = [
        [start_per_cam[i] + t for t in range(n_frames_raw)] for i in range(n_cameras)
    ]
    return (
        X_opt,
        cov,
        frame_indices,
        pixels_for_draw,
        frame_indices_all,
        positions_all_frames,
        detected_all_frames,
        pixels,
    )


def plot_3d_trajectory(X_opt, cov=None, out_path="trajectory_3d.png"):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("Install matplotlib to plot: pip install matplotlib")
        return
    n = len(X_opt)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    #mask = X_opt[:, 0] < 10
    #X_plot = X_opt[mask]
    #frame_index = np.arange(n)[mask]
    X_plot = X_opt
    frame_index = np.arange(n)
    sc = ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        X_plot[:, 2],
        c=frame_index,
        cmap="viridis",
        s=200,
        edgecolors="none",
    )
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
    ax.axvline(x=dimensions[0], color="r", linestyle="--")
    ax.axhline(y=dimensions[1], color="r", linestyle="--")
    ax.axvline(x=0, color="r", linestyle="--")
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlim(0, dimensions[0])
    ax.set_ylim(0, dimensions[1])
    ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Frame", fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.show()


def save_frames(video_paths, frame_indices, out_dir, pixels=None, detected=None, box_half_size=30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cameras = len(video_paths)
    n_frames = len(frame_indices[0])

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
                raise RuntimeError(
                    f"Could not read frame {frame_indices[i][t]} from video {i}."
                )
            h, w = frame.shape[:2]
            font_scale = max(0.6, min(2.5, w / 600.0))
            thickness = max(2, int(round(w / 500)))
            has_position = (
                pixels is not None
                and i < len(pixels)
                and t < len(pixels[i])
                and pixels[i][t] is not None
            )
            is_real_detection = (
                detected is not None
                and i < len(detected)
                and t < len(detected[i])
                and detected[i][t]
            )
            if has_position and (is_real_detection if detected is not None else True):
                cx = float(pixels[i][t][0])
                cy = float(pixels[i][t][1])
                box_half_w = max(15, int(round(box_half_size)))
                box_half_h = max(15, int(round(box_half_size)))
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
                cx = float(pixels[i][t][0])
                cy = float(pixels[i][t][1])
                pt = (int(round(cx)), int(round(cy)))
                marker_size = max(24, int(round(box_half_size * 1.3)))
                cv2.drawMarker(frame, pt, (0, 165, 255), cv2.MARKER_CROSS, marker_size, thickness)
                status_text = "Predicted (no detection)"
                status_color = (0, 165, 255)
            else:
                status_text = "No bounding box found"
                status_color = (0, 0, 255)
            cv2.putText(
                frame,
                f"Cam {i}: {status_text}",
                (20, 40 + int(40 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                status_color,
                thickness,
            )
            if status_text == "No bounding box found":
                (tw, th), _ = cv2.getTextSize(
                    status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                tx = (w - tw) // 2
                ty = (h + th) // 2
                cv2.putText(
                    frame,
                    status_text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    status_color,
                    thickness,
                )
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
    if not cam_0_pos or not cam_1_pos:
        raise ValueError(
            "Fill cam_0_pos and cam_1_pos (same number of x,y frames; native video resolution)."
        )

    start_time = time.perf_counter()
    video_paths = [
        "Video_Camera_Processing/throws/Arushi_throw_0.mp4",
        "Video_Camera_Processing/throws/Arushi_throw_1.mp4",
    ]
    P_list_path = "Video_Camera_Processing/P_list.npy"
    dt = 1.0 / 30.0
    g = [0.0, 0.0, -9.81]
    pixel_sigma = 1.0
    physics_sigma = 0.1
    omega_phys = 0.0
    out_path = "trajectory_3d.png"
    side_by_side_dir = _PROJECT_ROOT / "sample_data" / "trajectory_side_by_side_manual"
    video_paths = [Path(p) for p in video_paths]
    for p in video_paths:
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")

    P_list = np.load(P_list_path)
    if P_list.ndim != 3 or P_list.shape[1] != 3 or P_list.shape[2] != 4:
        raise ValueError("P_list must have shape (n_cameras, 3, 4)")
    P_list = [P_list[i] for i in range(len(P_list))]

    orig_sizes = [read_video_frame_size(p) for p in video_paths]
    print(f"orig_sizes (width, height) from videos: {orig_sizes}")

    (X_opt, cov, frame_indices, pixels_for_draw, frame_indices_all, positions_all_frames, detected_all_frames, pixels_full) = run_pipeline_manual(
        cam_0_pos,
        cam_1_pos,
        P_list=P_list,
        orig_sizes=orig_sizes,
        dt=dt,
        g=g,
        pixel_sigma=pixel_sigma,
        physics_sigma=physics_sigma,
        omega_phys=omega_phys,
    )

    print(f"Estimated 3D trajectory: {len(X_opt)} frames, dt={dt} s")
    print(f"  x range: [{X_opt[:, 0].min():.3f}, {X_opt[:, 0].max():.3f}] m")
    print(f"  y range: [{X_opt[:, 1].min():.3f}, {X_opt[:, 1].max():.3f}] m")
    print(f"  z range: [{X_opt[:, 2].min():.3f}, {X_opt[:, 2].max():.3f}] m")

    for i in range(len(X_opt)):
        print("--------------------------------")
        print(f"Frame {i}")
        print(f"Pixel (full res): {pixels_full[0][i]}, {pixels_full[1][i]}")
        print(f"Position: {X_opt[i]}")
        print("--------------------------------")

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if can_plot_3d:
        plot_3d_trajectory(X_opt, cov=cov, out_path=out_path)

    if can_save_frames:
        for i in range(len(pixels_full)):
            plot_2d_pixels(
                pixels_full[i], dimensions=orig_sizes[i], out_path=f"trajectory_2d_manual_{i}.png"
            )
    if can_save_frames:
        save_frames(
            video_paths,
            frame_indices_all,
            side_by_side_dir,
            pixels=positions_all_frames,
            detected=detected_all_frames,
        )


if __name__ == "__main__":
    main()
