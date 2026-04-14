import os
import cv2
import numpy as np

from CameraIntrinsicsCalibration import calibrate_cameras_and_save
from CameraPositionCalibration_Checkerboard import estimate_camera_pos, visualize_camera_layout

P_list_path = "Video_Camera_Processing/P_list.npy"
K_list_path = "Video_Camera_Processing/K_list.npy"
dist_list_path = "Video_Camera_Processing/dist_list.npy"
R_list_path = "Video_Camera_Processing/R_list.npy"
t_list_path = "Video_Camera_Processing/t_list.npy"
camera_positions_path = "Video_Camera_Processing/camera_positions.npy"
calibration_output_dir = "Video_Camera_Processing"


def make_P_list(
    image_list_dir,
    calibration_folders=None,
    intrinsics_list=None,
    dist_coeffs_list=None,
    CHECKERBOARD=None,
    square_size_m=None,
    visualize=False,
    boards=None,
    overwrite=False,
    extrinsics_debug_draw=False,
):
    """
    boards: list length num_cameras; each entry is a list of checkerboard dicts for
    CameraPositionCalibration_Checkerboard.estimate_camera_pos.
    """
    num_cameras = len(image_list_dir)

    if intrinsics_list is None or dist_coeffs_list is None or overwrite:
        if calibration_folders is None:
            calibration_folders = [
                f"Video_Camera_Processing/intrinsics_calibration_camera_{i}"
                for i in range(num_cameras)
            ]
        calibrate_cameras_and_save(
            calibration_folders,
            output_dir=calibration_output_dir,
            K_list_filename=os.path.basename(K_list_path),
            dist_list_filename=os.path.basename(dist_list_path),
            checkerboard=CHECKERBOARD,
            square_size_m=square_size_m,
        )
    else:
        print("Using provided intrinsics and dist_coeffs")

    K_list_arr = np.load(K_list_path, allow_pickle=True)
    dist_list_arr = np.load(dist_list_path, allow_pickle=True)
    intrinsics_list = [K_list_arr[i] for i in range(num_cameras)]
    dist_coeffs_list = [dist_list_arr[i] for i in range(num_cameras)]

    P_list = []
    K_list_out = []
    R_list = []
    t_list = []
    rvec_list = []
    camera_positions = []

    for i in range(num_cameras):
        image = cv2.imread(image_list_dir[i])
        if image is None:
            raise ValueError(f"Image {image_list_dir[i]} not found")
        intrinsics = intrinsics_list[i]
        dist_coeffs = dist_coeffs_list[i]
        rvec, tvec, camera_position = estimate_camera_pos(
            image,
            boards[i],
            intrinsics,
            dist_coeffs,
            debug_draw=extrinsics_debug_draw,
            debug_window_prefix=f"cam{i}",
        )
        R, _ = cv2.Rodrigues(rvec)
        P = intrinsics @ np.hstack([R, tvec])
        P_list.append(P)
        K_list_out.append(intrinsics)
        R_list.append(R)
        t_list.append(tvec)
        rvec_list.append(rvec)
        camera_positions.append(camera_position)

    np.save(P_list_path, P_list)
    np.save(K_list_path, K_list_out)
    np.save(R_list_path, R_list)
    np.save(t_list_path, t_list)
    np.save(camera_positions_path, camera_positions)
    print(f"Saved P_list: {P_list_path}")
    print(f"Saved K_list: {K_list_path}")
    print(f"Saved R_list: {R_list_path}")
    print(f"Saved t_list: {t_list_path}")
    print(f"Saved camera_positions: {camera_positions_path}")

    if visualize:
        cam_positions_arr = np.array(
            [cp.reshape(3) for cp in camera_positions], dtype=np.float64
        )
        rvecs_arr = np.array([rv.reshape(3) for rv in rvec_list], dtype=np.float64)
        visualize_camera_layout(
            cam_positions_arr, rvecs_arr, origin=(0.0, 0.0, 0.0), axis_length=0.5
        )


if __name__ == "__main__":
    image_list_dir = [
        "Video_Camera_Processing/extrinsics_calibration/extrinsics_0.png",
        "Video_Camera_Processing/extrinsics_calibration/extrinsics_1.png",
    ]

    # Intrinsics
    square_size_m = 0.015
    CHECKERBOARD = (10, 7)

    # Extrinsics
    ext_cb = (7, 10)
    ext_square_m = 0.015

    boards = [
        [
            {
            "checkerboard": ext_cb,
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
            "checkerboard": ext_cb,
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
            "checkerboard": ext_cb,
            "square_size_m": square_size_m,
            "R": np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64),
            "t": np.array([[7.16045777039], [-2.60619349214], [23.0 / 100]], dtype=np.float64),
            "corner_order_fix": "none",
        },
        ],
        [
            {
            "checkerboard": ext_cb,
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
            "checkerboard": ext_cb,
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
        "checkerboard": ext_cb,
            "square_size_m": square_size_m,
            "R": np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64),
            "t": np.array([[7.16045777039], [-2.60619349214], [23.0 / 100]], dtype=np.float64),
            "corner_order_fix": "none",
        },
        ],
    ]

    calibration_folders = [
        "Video_Camera_Processing/intrinsics_calibration_camera_0",
        "Video_Camera_Processing/intrinsics_calibration_camera_1",
    ]
    intrinsics_list = np.load("Video_Camera_Processing/K_list.npy", allow_pickle=True)
    dist_coeffs_list = np.load("Video_Camera_Processing/dist_list.npy", allow_pickle=True)

    visualize = True
    overwrite = False
    extrinsics_debug_draw = False

    make_P_list(
        image_list_dir,
        calibration_folders=calibration_folders,
        intrinsics_list=intrinsics_list,
        dist_coeffs_list=dist_coeffs_list,
        CHECKERBOARD=CHECKERBOARD,
        square_size_m=square_size_m,
        visualize=visualize,
        boards=boards,
        overwrite=overwrite,
        extrinsics_debug_draw=extrinsics_debug_draw,
    )
