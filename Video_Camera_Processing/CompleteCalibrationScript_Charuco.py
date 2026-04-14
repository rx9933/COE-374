import os
import numpy as np
import cv2
from CameraPositionCalibration_Charuco import (create_board, estimate_camera_pos, visualize_camera_layout)
from CameraIntrinsicsCalibration import calibrate_cameras_and_save

P_list_path = "Video_Camera_Processing/P_list.npy"
K_list_path = "Video_Camera_Processing/K_list.npy"
dist_list_path = "Video_Camera_Processing/dist_list.npy"
R_list_path = "Video_Camera_Processing/R_list.npy"
t_list_path = "Video_Camera_Processing/t_list.npy"
plot_path = "Video_Camera_Processing/camera_poses.png"
camera_positions_path = "Video_Camera_Processing/camera_positions.npy"
calibration_output_dir = "Video_Camera_Processing"

def make_P_list(image_list_dir, calibration_folders=None, intrinsics_list=None, dist_coeffs_list=None, CHECKERBOARD=None, square_size_m=None, visualize=False, boards=None, overwrite=False):
    """
    Make P_list (projection matrices) for each camera. If intrinsics and dist_coeffs are not provided, run calibration and save to disk.
    Args:
        image_list_dir: List of image directories.
        marker_length: Length of the marker in meters.
        marker_world_positions_list: List of marker world positions.
        calibration_folders: List of calibration folders.
        intrinsics_list: List of intrinsics.
        dist_coeffs_list: List of distortion coefficients.
    Returns:
        P_list: List of projection matrices.
        K_list: List of intrinsics.
        R_list: List of rotation matrices.
        t_list: List of translation vectors.
        camera_positions: List of camera positions.
    """
    num_cameras = len(image_list_dir)

    if intrinsics_list is None or dist_coeffs_list is None or overwrite:
        if calibration_folders is None:
            calibration_folders = [f"Video_Camera_Processing/intrinsics_calibration_camera_{i}" for i in range(num_cameras)]
        calibrate_cameras_and_save(calibration_folders, output_dir=calibration_output_dir, K_list_filename=os.path.basename(K_list_path), dist_list_filename=os.path.basename(dist_list_path), checkerboard=CHECKERBOARD, square_size_m=square_size_m)
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
        rvec, tvec, camera_position = estimate_camera_pos(image, boards[i], intrinsics, dist_coeffs)
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
        cam_positions_arr = np.array([cp.reshape(3) for cp in camera_positions], dtype=np.float64)
        rvecs_arr = np.array([rv.reshape(3) for rv in rvec_list], dtype=np.float64)
        visualize_camera_layout(cam_positions_arr, rvecs_arr, origin=(0.0, 0.0, 0.0), axis_length=0.5)


if __name__ == "__main__":
    #image_list_dir = ["Video_Camera_Processing/extrinsics_calibration/extrinsics_0.png", "Video_Camera_Processing/extrinsics_calibration/extrinsics_1.png"]
    image_list_dir = ["Video_Camera_Processing/extrinsics_calibration/extrinsics1.jpeg"]
    #Intrinsics
    square_size_m = 0.025
    CHECKERBOARD = (9, 6)


    #Extrinsics
    squares_x = 7
    squares_y = 10
    square_length = 0.015
    marker_length = 0.011

    board, aruco_dict = create_board(squares_x=squares_x, squares_y=squares_y, square_length=square_length, marker_length=marker_length)
    boards = [
        [ #camera 0
            {
                "board": board,
                "aruco_dict": aruco_dict,
                "R": np.eye(3),
                "t": np.array([[0], [0], [0]])
            }
        ],
        
        [ #camera 1
                {
                "board": board,
                "aruco_dict": aruco_dict,
                "R": np.eye(3),
                "t": np.array([[0], [0], [0]])
            }
        ]
    ]

    calibration_folders = ["Video_Camera_Processing/intrinsics_calibration_camera_0", "Video_Camera_Processing/intrinsics_calibration_camera_1"]
    intrinsics_list = "Video_Camera_Processing/K_list.npy"
    dist_coeffs_list = "Video_Camera_Processing/dist_list.npy"

    visualize = True
    overwrite = False
    make_P_list(image_list_dir, calibration_folders=calibration_folders, intrinsics_list=intrinsics_list, dist_coeffs_list=dist_coeffs_list, CHECKERBOARD=CHECKERBOARD, square_size_m=square_size_m, visualize=visualize, boards=boards, overwrite=overwrite)
