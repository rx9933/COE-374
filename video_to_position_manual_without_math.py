"""
Video to 3D position — manual pixel coordinates (native resolution), same inputs as
video_to_position_manual.py, but 3D points come from cv2.triangulatePoints only
(no physics / scipy trajectory optimization).

Optionally decomposes each 3x4 P with cv2.decomposeProjectionMatrix to sanity-check
K, R, and t implied by P (nonsensical intrinsics/extrinsics often show up here).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

import video_to_position_manual as vtm

can_save_frames = True
can_plot_3d = True


def _P_to_float34(P):
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (3, 4):
        raise ValueError("Each P must be shape (3, 4), got %s" % (P.shape,))
    return P


def report_decompose_projection_matrices(P_list, prefix=""):
    """
    For each 3x4 projection matrix P, run cv2.decomposeProjectionMatrix.
    P = K @ [R | t] up to scale; bad calibrations often yield negative focal lengths,
    extreme principal points, or non-orthogonal R.
    """
    for i, P in enumerate(P_list):
        P = _P_to_float34(P)
        out = cv2.decomposeProjectionMatrix(P)
        # OpenCV Python:7-tuple (no leading retval): K, R, t(4x1), rotX, rotY, rotZ, euler(3x1)
        # Some builds return 8-tuple with retval first; handle both.
        first = np.asarray(out[0])
        off = 0 if (first.ndim == 2 and first.shape == (3, 3)) else 1
        cam_mat = np.asarray(out[off + 0], dtype=np.float64).reshape(3, 3)
        rot_mat = np.asarray(out[off + 1], dtype=np.float64).reshape(3, 3)
        trans_v = np.asarray(out[off + 2], dtype=np.float64).reshape(-1, 1)
        euler = np.asarray(out[off + 6], dtype=np.float64).ravel() if len(out) > off + 6 else None
        fx, fy = cam_mat[0, 0], cam_mat[1, 1]
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]
        skew = cam_mat[0, 1]
        det_r = float(np.linalg.det(rot_mat))
        ortho_err = float(np.linalg.norm(rot_mat.T @ rot_mat - np.eye(3), ord="fro"))
        label = "%scamera %d" % (prefix + " " if prefix else "", i)
        print("--- decomposeProjectionMatrix (%s) ---" % label)
        print("  K (from decomposition):\n%s" % cam_mat)
        print("  fx=%.4f fy=%.4f cx=%.4f cy=%.4f skew=%.6f" % (fx, fy, cx, cy, skew))
        print("  R (3x3):\n%s" % rot_mat)
        print("  det(R)=%.6f  ||R^T R - I||_F=%.6g" % (det_r, ortho_err))
        print("  t (translation / homogeneous, from OpenCV):\n%s" % trans_v.ravel())
        if euler is not None and euler.size >= 3:
            print("  euler (deg): roll=%.3f pitch=%.3f yaw=%.3f" % tuple(euler.ravel()[:3]))
        else:
            print("  euler: (not returned by this OpenCV build)")
        if fx <= 0 or fy <= 0:
            print("  WARNING: non-positive focal length from decomposition.")
        if abs(det_r - 1.0) > 0.05 or ortho_err > 0.05:
            print("  WARNING: R is far from a proper rotation.")


def _triangulate_opencv_two_view(P0, P1, pixels0, pixels1):
    """pixels*: list of length N with (u, v) per frame. Returns (N, 3) in world coords."""
    n = len(pixels0)
    if len(pixels1) != n:
        raise ValueError("pixel lists must match length")
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts0 = np.zeros((2, n), dtype=np.float64)
    pts1 = np.zeros((2, n), dtype=np.float64)
    for k in range(n):
        pts0[0, k] = float(pixels0[k][0])
        pts0[1, k] = float(pixels0[k][1])
        pts1[0, k] = float(pixels1[k][0])
        pts1[1, k] = float(pixels1[k][1])
    P0 = _P_to_float34(P0)
    P1 = _P_to_float34(P1)
    X_h = cv2.triangulatePoints(P0, P1, pts0, pts1)
    X_h = np.asarray(X_h, dtype=np.float64)
    if X_h.shape != (4, n):
        raise RuntimeError("triangulatePoints expected shape (4, N), got %s" % (X_h.shape,))
    w = X_h[3, :]
    bad = np.abs(w) < 1e-12
    if np.any(bad):
        print("WARNING: %d / %d points have |w| < 1e-12 in homogeneous triangulation." % (np.sum(bad), n))
    Xw = (X_h[:3, :] / w).T
    return Xw


def run_pipeline_manual(
    cam_0_pos,
    cam_1_pos,
    P_list,
    orig_sizes,
    dt=None,
    g=None,
    pixel_sigma=None,
    physics_sigma=None,
    omega_phys=None,
    decompose_P=True,
):
    """
    Same preprocessing and return layout as video_to_position_manual.run_pipeline_manual.
    3D points: per-frame cv2.triangulatePoints(P0, P1, u0, v0, u1, v1).

    dt, g, pixel_sigma, physics_sigma, omega_phys are accepted for API compatibility
    and ignored (no trajectory optimization).

    orig_sizes is validated for length2 (same as original) but not used in triangulation;
    keep it for callers that pass video dimensions.

    If decompose_P, prints decomposeProjectionMatrix diagnostics for each P.
    """
    n_cameras = 2
    if len(P_list) != n_cameras or len(orig_sizes) != n_cameras:
        raise ValueError("Expected 2 cameras: P_list and orig_sizes length 2.")

    P_list = [_P_to_float34(P_list[i]) for i in range(n_cameras)]
    if decompose_P:
        report_decompose_projection_matrices(P_list)

    pairs0 = vtm._xy_pairs_from_cam_pos(cam_0_pos)
    pairs1 = vtm._xy_pairs_from_cam_pos(cam_1_pos)
    n0, n1 = len(pairs0), len(pairs1)
    if n0 != n1:
        raise ValueError("cam_0_pos has %s (x,y) frames, cam_1_pos has %s; counts must match." % (n0, n1))

    n_frames_raw = n0
    start_per_cam = [0, 0]
    all_pairs = [pairs0, pairs1]

    positions_full = []
    for i in range(n_cameras):
        row = []
        for t in range(n_frames_raw):
            x, y = all_pairs[i][t]
            row.append(vtm._point_from_xy(x, y))
        positions_full.append(row)

    valid_t = [t for t in range(n_frames_raw) if all(positions_full[i][t] is not None for i in range(n_cameras))]

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

    if n_frames < 1:
        raise ValueError("Need at least 1 frame with points on both cameras; got %s after filtering." % n_frames)

    if n_frames < n_frames_raw:
        valid_set = set(valid_t)
        dropped_common_t = [t for t in range(n_frames_raw) if t not in valid_set]
        print("Dropped %s frames with missing manual points; using %s frames." % (n_frames_raw - n_frames, n_frames))
        print("  Aligned-window indices dropped (0 .. %s): %s" % (n_frames_raw - 1, dropped_common_t))
        print("  Original 0-based video frame index per camera for those slots:")
        for i in range(n_cameras):
            print("    camera %s: %s" % (i, [start_per_cam[i] + t for t in dropped_common_t]))
        for t in dropped_common_t:
            missing = [j for j in range(n_cameras) if positions_full[j][t] is None]
            print("    t=%s -> missing cameras %s" % (t, missing))

    pixels_for_draw = [[positions_kept[i][k].copy() for k in range(n_frames)] for i in range(n_cameras)]
    pixels = [[positions_kept[i][k].copy() for k in range(n_frames)] for i in range(n_cameras)]

    X_opt = _triangulate_opencv_two_view(P_list[0], P_list[1], pixels[0], pixels[1])
    cov = None

    frame_indices = [[start_per_cam[i] + valid_t[t] for t in range(n_frames)] for i in range(n_cameras)]
    frame_indices_all = [[start_per_cam[i] + t for t in range(n_frames_raw)] for i in range(n_cameras)]
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


def main():
    if not vtm.cam_0_pos or not vtm.cam_1_pos:
        raise ValueError("Fill video_to_position_manual cam_0_pos and cam_1_pos (native resolution).")

    start_time = time.perf_counter()
    video_paths = [
        _PROJECT_ROOT / "Video_Camera_Processing/throws/Arushi_throw_0.mp4",
        _PROJECT_ROOT / "Video_Camera_Processing/throws/Arushi_throw_1.mp4",
    ]
    P_list_path = _PROJECT_ROOT / "Video_Camera_Processing/P_list.npy"
    dt = 1.0 / 30.0
    g = [0.0, 0.0, -9.81]
    out_path = _PROJECT_ROOT / "trajectory_3d_opencv_triangulate.png"
    side_by_side_dir = _PROJECT_ROOT / "sample_data" / "trajectory_side_by_side_manual_opencv"

    for p in video_paths:
        if not p.exists():
            raise FileNotFoundError("Video not found: %s" % p)
    if not P_list_path.exists():
        raise FileNotFoundError("Missing %s" % P_list_path)

    P_list = np.load(P_list_path)
    if P_list.ndim != 3 or P_list.shape[1] != 3 or P_list.shape[2] != 4:
        raise ValueError("P_list must have shape (n_cameras, 3, 4)")
    P_list = [P_list[i] for i in range(len(P_list))]

    orig_sizes = [vtm.read_video_frame_size(p) for p in video_paths]
    print("orig_sizes (width, height) from videos: %s" % orig_sizes)

    (X_opt, cov, frame_indices, pixels_for_draw, frame_indices_all, positions_all_frames, detected_all_frames, pixels_full) = run_pipeline_manual(
        vtm.cam_0_pos,
        vtm.cam_1_pos,
        P_list=P_list,
        orig_sizes=orig_sizes,
        dt=dt,
        g=g,
        decompose_P=True,
    )

    print("OpenCV triangulatePoints: %s frames (no physics smoothing)" % len(X_opt))
    print("  x range: [%.3f, %.3f] m" % (X_opt[:, 0].min(), X_opt[:, 0].max()))
    print("  y range: [%.3f, %.3f] m" % (X_opt[:, 1].min(), X_opt[:, 1].max()))
    print("  z range: [%.3f, %.3f] m" % (X_opt[:, 2].min(), X_opt[:, 2].max()))

    for i in range(len(X_opt)):
        print("--------------------------------")
        print("Frame %s" % i)
        print("Pixel (full res): %s, %s" % (pixels_full[0][i], pixels_full[1][i]))
        print("Position: %s" % X_opt[i])
        print("--------------------------------")

    print("Time taken: %.2f seconds" % (time.perf_counter() - start_time,))

    if can_plot_3d:
        vtm.plot_3d_trajectory(X_opt, cov=cov, out_path=str(out_path))

    if can_save_frames:
        for i in range(len(pixels_full)):
            vtm.plot_2d_pixels(
                pixels_full[i],
                dimensions=orig_sizes[i],
                out_path=str(_PROJECT_ROOT / ("trajectory_2d_manual_opencv_%s.png" % i)),
            )
        vtm.save_frames(
            video_paths,
            frame_indices_all,
            side_by_side_dir,
            pixels=positions_all_frames,
            detected=detected_all_frames,
        )


if __name__ == "__main__":
    main()
