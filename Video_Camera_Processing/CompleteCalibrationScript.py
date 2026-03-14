import os
import numpy as np
import cv2
from CameraPositionCalibration import detect_markers, build_world_points, estimate_camera_pose
from CameraIntrinsicsCalibration import calibrate_cameras_and_save

P_list_path = "Video_Camera_Processing/P_list.npy"
K_list_path = "Video_Camera_Processing/K_list.npy"
dist_list_path = "Video_Camera_Processing/dist_list.npy"
R_list_path = "Video_Camera_Processing/R_list.npy"
t_list_path = "Video_Camera_Processing/t_list.npy"
plot_path = "Video_Camera_Processing/camera_poses.png"
camera_positions_path = "Video_Camera_Processing/camera_positions.npy"
calibration_output_dir = "Video_Camera_Processing"

def get_camera_pose(image, marker_length, marker_world_positions, intrinsics, dist_coeffs):
    corners, ids = detect_markers(image)
    if corners is None:
        return None, None, None
    object_points, image_points = build_world_points(corners, ids, marker_length, marker_world_positions)
    rvec, tvec, camera_position = estimate_camera_pose(object_points, image_points, intrinsics, dist_coeffs)
    return rvec, tvec, camera_position


def make_P_list(image_list_dir, marker_length, marker_world_positions_list, calibration_folders=None, intrinsics_list=None, dist_coeffs_list=None):
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

    # Try loading existing calibration from disk first
    # if not found, run calibration and save to disk
    if intrinsics_list is None or dist_coeffs_list is None:
        if intrinsics_list is None or dist_coeffs_list is None:
            if calibration_folders is None:
                calibration_folders = [f"calibration_camera_{i}" for i in range(num_cameras)]
            calibrate_cameras_and_save(calibration_folders, output_dir=calibration_output_dir, K_list_filename=os.path.basename(K_list_path), dist_list_filename=os.path.basename(dist_list_path))
        K_list_arr = np.load(K_list_path, allow_pickle=True)
        dist_list_arr = np.load(dist_list_path, allow_pickle=True)
        intrinsics_list = [K_list_arr[i] for i in range(num_cameras)]
        dist_coeffs_list = [dist_list_arr[i] for i in range(num_cameras)]


    P_list = []
    K_list_out = []
    R_list = []
    t_list = []
    camera_positions = []
    for i in range(num_cameras):
        image = cv2.imread(image_list_dir[i])
        if image is None:
            raise ValueError(f"Image {image_list_dir[i]} not found")
        marker_world_positions = marker_world_positions_list[i]
        intrinsics = intrinsics_list[i]
        dist_coeffs = dist_coeffs_list[i]
        rvec, tvec, camera_position = get_camera_pose(image, marker_length, marker_world_positions, intrinsics, dist_coeffs)
        R, _ = cv2.Rodrigues(rvec)
        P = intrinsics @ np.hstack([R, tvec])
        P_list.append(P)
        K_list_out.append(intrinsics)
        R_list.append(R)
        t_list.append(tvec)
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


if __name__ == "__main__":
    image_list_dir = ["Video_Camera_Processing/camera_0", "Video_Camera_Processing/camera_1"]
    marker_length = 0.20
    marker_world_positions_list = [
        {
            0: (0.0, 0.0, 0.0),
            1: (2.0, 0.0, 0.0),
            2: (0.0, 2.0, 0.0),
            3: (2.0, 2.0, 0.0),
            4: (1.0, 3.0, 0.0)
        }
    ] * 2
    calibration_folders = ["Video_Camera_Processing/camera_0", "Video_Camera_Processing/camera_1"]
    intrinsics_list = None
    dist_coeffs_list = None
    make_P_list(image_list_dir, marker_length, marker_world_positions_list, calibration_folders=calibration_folders, intrinsics_list=intrinsics_list, dist_coeffs_list=dist_coeffs_list)