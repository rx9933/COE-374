import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9, 6)
square_size = 0.025  # meters


def calibrate_camera_from_folder(folder_path, checkerboard=CHECKERBOARD, square_size_m=square_size):
    folder_path = os.path.abspath(folder_path)
    pattern = os.path.join(folder_path, "*.jpg")
    images = glob.glob(pattern)

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    gray_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    print(f"  Reprojection error: {ret:.4f}")
    if ret > 1.0:
        raise ValueError(f"Reprojection error is too high: {ret:.4f}, try to use a different checkerboard or square size or increase the number of images")

    return K, dist


def calibrate_cameras_and_save(calibration_folders, output_dir=".", K_list_filename="K_list.npy", dist_list_filename="dist_list.npy", checkerboard=CHECKERBOARD, square_size_m=square_size):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    K_list = []
    dist_list = []

    for i, folder in enumerate(calibration_folders):
        print(f"Calibrating camera {i} from folder: {folder}")
        K, dist = calibrate_camera_from_folder(folder, checkerboard=checkerboard, square_size_m=square_size_m)
        K_list.append(K)
        dist_list.append(dist)

    K_list_path = os.path.join(output_dir, K_list_filename)
    dist_list_path = os.path.join(output_dir, dist_list_filename)

    np.save(K_list_path, np.array(K_list, dtype=object))
    np.save(dist_list_path, np.array(dist_list, dtype=object))

    print(f"Saved {K_list_path} and {dist_list_path}")
    return K_list, dist_list


if __name__ == "__main__":
    calibration_folders = ["calibration_camera_0", "calibration_camera_1"]
    output_dir = "Video_Camera_Processing"
    calibrate_cameras_and_save(calibration_folders, output_dir=output_dir)
