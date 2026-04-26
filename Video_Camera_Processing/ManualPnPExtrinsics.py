import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
K_LIST_PATH = BASE_DIR / "K_list.npy"
DIST_LIST_PATH = BASE_DIR / "dist_list.npy"
R_LIST_PATH = BASE_DIR / "R_list.npy"
T_LIST_PATH = BASE_DIR / "t_list.npy"
P_LIST_PATH = BASE_DIR / "P_list.npy"
PLOT_PATH = BASE_DIR / "manual_pnp_extrinsics_poses.png"
SAVE_PLOT = True
SHOW_PLOT = True

world_points_list = [
    np.array(
        [
        [0, 0, 0], #origin
        [6.7818, -1.2192, 1.143], #tall
        [8.4582, -0.9144, 0.2032], #close
        [8.4836, -1.8288,  0.19685], #far
        ],
        dtype=np.float64,
    ),
    np.array(
        [
        [8.0518, 0.8636, 1.143], #tall
        [10.45464, 0, 0.2032], #right
        [9.4488, 2.2224999999999997, 0.19685], #left
        ],
        dtype=np.float64,
    ),
]

image_points_list = [
    np.array(
        [
        [424 , 1968],
        [1298 , 1629],
        [2516 , 2298],
        [1995 , 2430],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
        [2508 , 1482],
        [710 , 2219],
        [2557 , 2394],
        ],
        dtype=np.float64,
    ),
]


def prepare_correspondences(obj_pts, img_pts):
    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    if obj_pts.shape[0] != img_pts.shape[0]:
        raise ValueError(
            f"world_points has {obj_pts.shape[0]} rows but image_points has {img_pts.shape[0]} rows."
        )

    finite_mask = np.isfinite(obj_pts).all(axis=1) & np.isfinite(img_pts).all(axis=1)
    nonzero_img_mask = np.linalg.norm(img_pts, axis=1) > 1e-12
    keep_mask = finite_mask & nonzero_img_mask

    obj_keep = obj_pts[keep_mask]
    img_keep = img_pts[keep_mask]
    n = obj_keep.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 valid correspondences after filtering; got {n}.")
    return obj_keep, img_keep


def solve_camera_from_manual_points(world_points, image_points, K, dist):
    world_use, image_use = prepare_correspondences(world_points, image_points)
    n_points = world_use.shape[0]
    if n_points == 3:
        pnp_flag = cv2.SOLVEPNP_SQPNP
    elif n_points < 6:
        pnp_flag = cv2.SOLVEPNP_EPNP
    else:
        pnp_flag = cv2.SOLVEPNP_ITERATIVE

    success, rvec, tvec = cv2.solvePnP(world_use, image_use, K, dist, flags=pnp_flag)
    if not success:
        raise RuntimeError("solvePnP failed")

    R_wc, _ = cv2.Rodrigues(rvec)
    cam_center_world = (-R_wc.T @ tvec).reshape(3)

    reproj, _ = cv2.projectPoints(world_use, rvec, tvec, K, dist)
    reproj = reproj.reshape(-1, 2)
    reproj_err = np.linalg.norm(reproj - image_use, axis=1)

    return R_wc, cam_center_world, reproj_err, pnp_flag, n_points


def make_projection_matrix(K, R_wc, cam_center_world):
    C = np.asarray(cam_center_world, dtype=np.float64).reshape(3, 1)
    tvec = -R_wc @ C
    return K @ np.hstack([R_wc, tvec])


def optical_axis_world(R_wc):
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    v = R_wc.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def plot_manual_extrinsics(
    R_stack,
    camera_centers_world,
    world_points_per_cam,
    output_path,
    arrow_scale=None,
    show=False,
):
    import matplotlib.pyplot as plt

    R_stack = np.asarray(R_stack, dtype=np.float64)
    camera_centers_world = np.asarray(camera_centers_world, dtype=np.float64)
    n = R_stack.shape[0]
    if camera_centers_world.shape[0] != n:
        raise ValueError("R_stack and camera_centers_world must have the same leading dimension.")

    if arrow_scale is None:
        span = float(np.linalg.norm(camera_centers_world, axis=1).max())
        arrow_scale = max(1.5, min(6.0, 0.25 * max(span, 1.0)))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_zlabel("Z (world)")

    ax.quiver(
        0, 0, 0, 1, 0, 0, color="r", arrow_length_ratio=0.15, linewidth=2.0, label="World +X"
    )
    ax.quiver(
        0, 0, 0, 0, 1, 0, color="g", arrow_length_ratio=0.15, linewidth=2.0, label="World +Y"
    )
    ax.quiver(
        0, 0, 0, 0, 0, 1, color="b", arrow_length_ratio=0.15, linewidth=2.0, label="World +Z"
    )

    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        c="black",
        s=500,
        marker="*",
        edgecolors="gold",
        linewidths=2.0,
        zorder=10,
        label="World origin (0, 0, 0)",
    )

    colors = ["orange", "cyan", "magenta", "tab:olive"]
    all_pts = [[0.0, 0.0, 0.0]]

    gcp_markers = ["o", "s", "^", "D"]
    for ci, wpts in enumerate(world_points_per_cam):
        wpts = np.asarray(wpts, dtype=np.float64).reshape(-1, 3)
        if wpts.size == 0:
            continue
        c = colors[ci % len(colors)]
        ax.scatter(
            wpts[:, 0],
            wpts[:, 1],
            wpts[:, 2],
            c=c,
            s=45,
            marker=gcp_markers[ci % len(gcp_markers)],
            alpha=0.85,
            label=f"PnP landmarks (cam {ci})",
            zorder=4,
        )
        all_pts.extend(wpts.tolist())

    for i in range(n):
        C = camera_centers_world[i]
        view_dir = optical_axis_world(R_stack[i])
        c = colors[i % len(colors)]
        all_pts.append(C.tolist())
        all_pts.append((C + view_dir * arrow_scale).tolist())

        ax.scatter(
            *C,
            color=c,
            s=140,
            marker="P",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=f"Cam {i} center",
        )
        ax.quiver(
            C[0],
            C[1],
            C[2],
            view_dir[0] * arrow_scale,
            view_dir[1] * arrow_scale,
            view_dir[2] * arrow_scale,
            color=c,
            arrow_length_ratio=0.18,
            linewidth=2.2,
            label=f"Cam {i}: optical axis (+Z)",
        )

    all_pts = np.asarray(all_pts, dtype=np.float64)
    margin = 1.2
    lo = all_pts.min(axis=0) - margin
    hi = all_pts.max(axis=0) + margin
    span = float(np.max(hi - lo))
    span = max(span, 3.0)
    center = 0.5 * (lo + hi)
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_title("Manual PnP extrinsics: origin, landmarks, camera centers, optical axis (+Z)")
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=140, bbox_inches="tight")
    print(f"Saved extrinsics visualization to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main():
    K_list = np.load(str(K_LIST_PATH), allow_pickle=True)
    dist_list = np.load(str(DIST_LIST_PATH), allow_pickle=True)

    n_cams = len(K_list)
    if len(world_points_list) != n_cams or len(image_points_list) != n_cams:
        raise ValueError(
            f"K_list has {n_cams} cameras, but world_points_list has {len(world_points_list)} "
            f"and image_points_list has {len(image_points_list)}."
        )

    R_list = []
    t_list = []
    P_list = []

    for cam_id in range(n_cams):
        K = np.asarray(K_list[cam_id], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dist_list[cam_id], dtype=np.float64)

        R_wc, cam_center_world, reproj_err, pnp_flag, n_points = solve_camera_from_manual_points(
            world_points_list[cam_id],
            image_points_list[cam_id],
            K,
            dist,
        )

        P = make_projection_matrix(K, R_wc, cam_center_world)
        R_list.append(R_wc)
        t_list.append(cam_center_world)
        P_list.append(P)

        print(f"\n=== Camera {cam_id} ===")
        print(f"Used {n_points} correspondences with flag={pnp_flag}")
        print("R:")
        print(R_wc)
        print("Camera center:")
        print(cam_center_world)
        print("Reprojection error per point (pixels):")
        print(reproj_err)
        print(f"Mean reprojection error: {float(np.mean(reproj_err)):.4f} px")

    R_list = np.stack(R_list, axis=0)
    t_list = np.stack(t_list, axis=0)
    P_list = np.stack(P_list, axis=0)

    np.save(str(R_LIST_PATH), R_list)
    np.save(str(T_LIST_PATH), t_list)
    np.save(str(P_LIST_PATH), P_list)

    print("\nSaved:")
    print(f"- {R_LIST_PATH}")
    print(f"- {T_LIST_PATH}")
    print(f"- {P_LIST_PATH}")

    if SAVE_PLOT:
        plot_manual_extrinsics(
            R_list,
            t_list,
            world_points_list,
            PLOT_PATH,
            show=SHOW_PLOT,
        )


if __name__ == "__main__":
    main()
