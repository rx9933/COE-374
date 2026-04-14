from __future__ import annotations

import os

import numpy as np

OUTPUT_DIR = "Video_Camera_Processing"
P_LIST_PATH = os.path.join(OUTPUT_DIR, "P_list.npy")
K_LIST_PATH = os.path.join(OUTPUT_DIR, "K_list.npy")
R_LIST_PATH = os.path.join(OUTPUT_DIR, "R_list.npy")
T_LIST_PATH = os.path.join(OUTPUT_DIR, "t_list.npy")
CAMERA_POSITIONS_PATH = os.path.join(OUTPUT_DIR, "camera_positions.npy")
PLOT_PATH = os.path.join(OUTPUT_DIR, "force_extrinsics_poses.png")

CAMERA_X_M = 14.538045600000002
CAMERA_Y_M_CAM0 = 4.572
CAMERA_Y_M_CAM1 = -4.572
CAMERA_Z_M = 1.3081
show_plot = True

TARGET_LOOK_AT_CAM0 = np.array([0.0, 0.0, 1.3081], dtype=np.float64)
TARGET_LOOK_AT_CAM1 = np.array([0.0, 0.0, 1.3081], dtype=np.float64)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def look_at(
    camera_position,
    target=np.array([0.0, 0.0, 0.0]),
    world_up=np.array([0.0, 0.0, 1.0]),
):
    """World rotation R: camera +Z axis points from camera toward target (make_P_list convention)."""
    C = np.asarray(camera_position, dtype=np.float64).ravel()
    T = np.asarray(target, dtype=np.float64).ravel()
    forward = T - C
    n = np.linalg.norm(forward)
    if n < 1e-10:
        raise ValueError("Camera position and target are too close.")
    forward = forward / n
    right = np.cross(forward, world_up)
    nr = np.linalg.norm(right)
    if nr < 1e-10:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / nr
    up = np.cross(right, forward)
    R = np.column_stack([right, up, forward]).T
    return R


def build_P_and_extrinsics(K, R, camera_center_world):
    """
    P = K @ [R | tvec] with X_cam = R @ X_world + tvec, camera center C = -R.T @ tvec.
    Same as make_P_list.make_P with t storing camera position C.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    C = np.asarray(camera_center_world, dtype=np.float64).reshape(3, 1)
    tvec = -R @ C
    P = K @ np.hstack([R, tvec])
    return P, tvec


def camera_center_and_view_from_opencv(R, tvec):
    """C = -R.T @ tvec; viewing axis +Z in camera frame -> world direction R.T @ e_z."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    C = (-R.T @ tvec.reshape(3, 1)).ravel()
    view_dir = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)
    return C, view_dir


def plot_extrinsics_layout(
    R_stack,
    tvec_stack,
    targets,
    output_path,
    arrow_scale=None,
    show=True,
):
    import matplotlib.pyplot as plt

    R_stack = np.asarray(R_stack, dtype=np.float64)
    tvec_stack = np.asarray(tvec_stack, dtype=np.float64)
    n = R_stack.shape[0]
    targets_list = [np.asarray(t, dtype=np.float64).ravel() for t in targets]
    if len(targets_list) != n:
        raise ValueError(f"targets length {len(targets_list)} must match number of cameras {n}")

    if arrow_scale is None:
        centers = []
        for i in range(n):
            C, _ = camera_center_and_view_from_opencv(R_stack[i], tvec_stack[i])
            centers.append(C)
        centers = np.array(centers, dtype=np.float64)
        span = float(np.linalg.norm(centers, axis=1).max())
        arrow_scale = max(2.0, min(8.0, 0.35 * span))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.quiver(0, 0, 0, 1, 0, 0, color="r", arrow_length_ratio=0.12, linewidth=1.5, label="world X")
    ax.quiver(0, 0, 0, 0, 1, 0, color="g", arrow_length_ratio=0.12, linewidth=1.5, label="world Y")
    ax.quiver(0, 0, 0, 0, 0, 1, color="b", arrow_length_ratio=0.12, linewidth=1.5, label="world Z")
    target_markers = ["*", "P", "X", "d"]
    for i, tgt in enumerate(targets_list):
        ax.scatter(
            *tgt,
            c="k",
            s=120,
            marker=target_markers[i % len(target_markers)],
            label=f"look-at cam {i}",
            zorder=5,
        )

    colors = ["orange", "cyan", "magenta", "yellow"]
    all_pts = [[0, 0, 0]] + [t.tolist() for t in targets_list]
    for i in range(n):
        R = R_stack[i]
        tvec = tvec_stack[i]
        C, view_dir = camera_center_and_view_from_opencv(R, tvec)
        c = colors[i % len(colors)]
        all_pts.append(C)
        all_pts.append(C + view_dir * arrow_scale)
        ax.scatter(*C, color=c, s=80, label=f"cam {i}")
        ax.quiver(
            C[0],
            C[1],
            C[2],
            view_dir[0] * arrow_scale,
            view_dir[1] * arrow_scale,
            view_dir[2] * arrow_scale,
            color=c,
            arrow_length_ratio=0.2,
            linewidth=2,
        )

    all_pts = np.array(all_pts, dtype=np.float64)
    margin = 1.5
    max_axis = float(np.abs(all_pts).max()) + margin
    max_axis = max(max_axis, 5.0)
    ax.set_xlim(-max_axis, max_axis)
    ax.set_ylim(-max_axis, max_axis)
    ax.set_zlim(-max_axis, max_axis)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("forceExtrinsics: camera centers and +Z toward per-camera look-at")
    plt.tight_layout()
    _plot_dir = os.path.dirname(os.path.abspath(output_path))
    if _plot_dir:
        os.makedirs(_plot_dir, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Saved 3D layout plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def load_K_list(path=None):
    path = path or K_LIST_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"K_list.npy not found at {path}; run intrinsics calibration first."
        )
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(arr[i], dtype=np.float64) for i in range(len(arr))]


def save_approximate_extrinsics(
    output_dir=None,
    k_list=None,
    camera_centers=None,
    targets=None,
    target=None,
    world_up=None,
    plot=True,
    plot_path=None,
    show_plot=True,
    verbose=True,
):
    output_dir = output_dir or OUTPUT_DIR
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    world_up = WORLD_UP if world_up is None else np.asarray(world_up, dtype=np.float64).ravel()

    if k_list is None:
        k_list = load_K_list(os.path.join(output_dir, os.path.basename(K_LIST_PATH)))

    if camera_centers is None:
        z = float(CAMERA_Z_M)
        camera_centers = [
            np.array([[CAMERA_X_M], [CAMERA_Y_M_CAM0], [z]], dtype=np.float64),
            np.array([[CAMERA_X_M], [CAMERA_Y_M_CAM1], [z]], dtype=np.float64),
        ]

    n_cams = len(camera_centers)
    if target is not None and targets is not None:
        raise ValueError("Pass at most one of target= (all cams) and targets= (per camera).")
    if target is not None:
        tt = np.asarray(target, dtype=np.float64).ravel()
        if tt.size != 3:
            raise ValueError("target= must be a length-3 world point")
        look_targets = [tt.copy() for _ in range(n_cams)]
    elif targets is None:
        if n_cams != 2:
            raise ValueError("Default per-camera targets require 2 cameras; pass targets= otherwise.")
        look_targets = [
            np.asarray(TARGET_LOOK_AT_CAM0, dtype=np.float64).ravel(),
            np.asarray(TARGET_LOOK_AT_CAM1, dtype=np.float64).ravel(),
        ]
    else:
        look_targets = [np.asarray(t, dtype=np.float64).ravel() for t in targets]
        if len(look_targets) != n_cams:
            raise ValueError(f"targets length {len(look_targets)} != number of cameras {n_cams}")

    P_list = []
    R_list = []
    t_list = []
    cam_pos_out = []

    for K, C, tgt in zip(k_list, camera_centers, look_targets):
        R = look_at(C.ravel(), target=tgt, world_up=world_up)
        P, tvec = build_P_and_extrinsics(K, R, C)
        P_list.append(P)
        R_list.append(R)
        t_list.append(tvec)
        cam_pos_out.append(-R.T @ tvec)

    p_path = os.path.join(output_dir, os.path.basename(P_LIST_PATH))
    r_path = os.path.join(output_dir, os.path.basename(R_LIST_PATH))
    t_path = os.path.join(output_dir, os.path.basename(T_LIST_PATH))
    cp_path = os.path.join(output_dir, os.path.basename(CAMERA_POSITIONS_PATH))

    np.save(p_path, np.stack(P_list, axis=0))
    np.save(r_path, np.stack(R_list, axis=0))
    np.save(t_path, np.stack(t_list, axis=0))
    np.save(cp_path, np.stack(cam_pos_out, axis=0))

    if verbose:
        print(f"Saved {p_path}  shape {np.stack(P_list).shape}")
        print(f"Saved {r_path}")
        print(f"Saved {t_path}  (OpenCV tvec for P = K @ [R|t])")
        print(f"Saved {cp_path}  (camera centers, should match input C)")

    plot_out = None
    if plot:
        fig_path = plot_path or os.path.join(output_dir, os.path.basename(PLOT_PATH))
        plot_out = plot_extrinsics_layout(
            np.stack(R_list, axis=0),
            np.stack(t_list, axis=0),
            look_targets,
            fig_path,
            show=show_plot,
        )

    return p_path, r_path, t_path, cp_path, plot_out


if __name__ == "__main__":
    save_approximate_extrinsics(plot=True, show_plot=show_plot)
