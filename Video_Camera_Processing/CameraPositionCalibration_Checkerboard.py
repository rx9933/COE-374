
from __future__ import annotations
import cv2
import numpy as np


def make_checkerboard_points(cb, square_size):
    cols, rows = cb
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def board_to_world_points(obj_pts_board, R_board, t_board):
    t = np.asarray(t_board, dtype=np.float64).reshape(3, 1)
    obj_pts_board = np.asarray(obj_pts_board, dtype=np.float64).reshape(-1, 3)
    obj_pts_world = (R_board @ obj_pts_board.T + t).T.astype(np.float32)
    return obj_pts_world


def reorder_checkerboard_objp(objp, checkerboard, fix):
    if fix is None or fix == "none":
        return np.asarray(objp, dtype=np.float32)
    cols, rows = int(checkerboard[0]), int(checkerboard[1])
    g = np.asarray(objp, dtype=np.float32).reshape(rows, cols, 3)
    if fix == "flip_lr":
        g = g[:, ::-1, :].copy()
    elif fix == "flip_ud":
        g = g[::-1, :, :].copy()
    elif fix == "rot180":
        g = g[::-1, ::-1, :].copy()
    else:
        raise ValueError(f"Unknown corner_order_fix: {fix!r}")
    return g.reshape(-1, 3)


def detect_checkerboard_corners(
    image,
    checkerboard,
    roi=None,
    use_sb=True,
    subpix_window=(11, 11),
):
    """
    roi: None or (x0, y0, x1, y1) in full-image coordinates (half-open like slicing).
    Returns (success, corners_Nx1x2_float32_or_None).
    """
    if roi is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in roi]
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return False, None
        offset_x, offset_y = x0, y0
    else:
        crop = image
        offset_x = offset_y = 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret = False
    corners = None
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        ret, corners = cv2.findChessboardCornersSB(
            gray, checkerboard, cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    if not ret or corners is None:
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, classic_flags)

    if not ret or corners is None:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(
        gray,
        corners,
        subpix_window,
        (-1, -1),
        criteria,
    )
    corners = corners.astype(np.float32)
    corners[:, 0, 0] += offset_x
    corners[:, 0, 1] += offset_y
    return True, corners


def draw_checkerboard_debug(image, checkerboard, corners, window_name="checkerboard_debug"):
    """Draws corners; index 0 is green, last is red — check consistency across boards."""
    if corners is None:
        return
    vis = image.copy()
    cv2.drawChessboardCorners(vis, checkerboard, corners, True)
    pts = corners.reshape(-1, 2)
    if len(pts) > 0:
        p0 = tuple(np.round(pts[0]).astype(int))
        p1 = tuple(np.round(pts[-1]).astype(int))
        cv2.circle(vis, p0, 8, (0, 255, 0), 2)
        cv2.putText(vis, "0", (p0[0] + 6, p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(vis, p1, 8, (0, 0, 255), 2)
        cv2.putText(vis, "last", (p1[0] + 6, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow(window_name, vis)
    cv2.waitKey(1)


def estimate_camera_pos(
    image,
    boards,
    camera_matrix,
    dist_coeffs,
    debug_draw=False,
    debug_window_prefix="cb",
):
    all_obj_points = []
    all_img_points = []

    for bi, board_data in enumerate(boards):
        cb = board_data["checkerboard"]
        sq = float(board_data["square_size_m"])
        Rb = np.asarray(board_data["R"], dtype=np.float64)
        tb = board_data["t"]
        roi = board_data.get("roi")
        fix = board_data.get("corner_order_fix", "none")
        use_sb = board_data.get("use_find_chessboard_sb", True)

        ok, img_corners = detect_checkerboard_corners(
            image, cb, roi=roi, use_sb=use_sb
        )
        if not ok:
            continue

        if debug_draw:
            draw_checkerboard_debug(
                image,
                cb,
                img_corners,
                window_name=f"{debug_window_prefix}_{bi}",
            )

        obj_local = make_checkerboard_points(cb, sq)
        obj_local = reorder_checkerboard_objp(obj_local, cb, fix)
        obj_world = board_to_world_points(obj_local, Rb, tb)
        img_pts = img_corners.reshape(-1, 2).astype(np.float32)

        all_obj_points.append(obj_world)
        all_img_points.append(img_pts)

    if len(all_obj_points) == 0:
        raise ValueError("No checkerboards detected")

    all_obj_points = np.vstack(all_obj_points).astype(np.float32)
    all_img_points = np.vstack(all_img_points).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        all_obj_points,
        all_img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec

    return rvec, tvec, camera_position


def visualize_camera_layout(camera_position, rvec, origin=(0.0, 0.0, 0.0), axis_length=0.5):
    import matplotlib.pyplot as plt

    pos = np.asarray(camera_position, dtype=np.float64)
    rvecs = np.asarray(rvec, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
        rvecs = rvecs.reshape(1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ox, oy, oz = origin
    ax.scatter([ox], [oy], [oz], c="black", s=80)
    ax.text(ox, oy, oz, "origin")

    for i in range(pos.shape[0]):
        p = pos[i]
        ax.scatter(p[0], p[1], p[2], c="blue", s=100)
        ax.text(p[0], p[1], p[2], f"cam{i}")

        R, _ = cv2.Rodrigues(rvecs[i].reshape(3, 1))
        direction = R.T @ np.array([0, 0, 1], dtype=np.float64)
        ax.quiver(
            p[0],
            p[1],
            p[2],
            direction[0],
            direction[1],
            direction[2],
            length=axis_length,
            color="green",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def main():
    image = cv2.imread("Video_Camera_Processing/extrinsics_calibration/extrinsics_0.png")
    if image is None:
        raise ValueError("Image not found")

    camera_matrix = np.load("Video_Camera_Processing/K_list.npy", allow_pickle=True)[0]
    dist_coeffs = np.load("Video_Camera_Processing/dist_list.npy", allow_pickle=True)[0]

    square_size_m = 0.015
    checkerboard = (11, 8)

    boards = [
        {
            "checkerboard": checkerboard,
            "square_size_m": square_size_m,
            "R": np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64),
            "t": np.array([[7.62], [0.0], [23.0 / 100]], dtype=np.float64),
            "corner_order_fix": "none",
        },
        {
            "checkerboard": checkerboard,
            "square_size_m": square_size_m,
            "R": np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64),
            "t": np.array([[7.26892107732], [2.28630408556], [23.0 / 100]], dtype=np.float64),
            "corner_order_fix": "none",
        },
        {
            "checkerboard": checkerboard,
            "square_size_m": square_size_m,
            "R": np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64),
            "t": np.array([[7.16045777039], [-2.60619349214], [23.0 / 100]], dtype=np.float64),
            "corner_order_fix": "none",
        },
    ]

    rvec, tvec, camera_position = estimate_camera_pos(
        image, boards, camera_matrix, dist_coeffs, debug_draw=False
    )

    print("\nCamera position:")
    print(camera_position.flatten())

    visualize_camera_layout(camera_position, rvec)


if __name__ == "__main__":
    main()
