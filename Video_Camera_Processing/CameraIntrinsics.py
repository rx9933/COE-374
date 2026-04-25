import glob
import os
import warnings

import cv2
import numpy as np

CHECKERBOARD = (7, 8)
square_size = 0.015  # meters

_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def get_images(folder_path):
    folder_path = os.path.abspath(folder_path)
    paths = []
    for g in _IMAGE_GLOBS:
        paths.extend(glob.glob(os.path.join(folder_path, g)))
    return sorted(set(os.path.abspath(p) for p in paths))


def find_checkerboard(objp, img_corners, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2).astype(np.float64)
    obs = img_corners.reshape(-1, 2).astype(np.float64)
    return float(np.sqrt(np.mean(np.sum((proj - obs) ** 2, axis=1))))


def determine_error(objpoints, imgpoints, ok_paths, K, dist, rvecs, tvecs):
    rows = []
    for i, path in enumerate(ok_paths):
        rms = find_checkerboard(objpoints[i], imgpoints[i], rvecs[i], tvecs[i], K, dist)
        rows.append((rms, path))
    rows.sort(key=lambda x: -x[0])
    return rows


def print_report(intrinsics_cases):
    print("  Per-image RMS reprojection (pixels), worst → best (review top rows for blur / glare / bad corners):")
    for rank, (rms, path) in enumerate(intrinsics_cases, start=1):
        print("    %3d  rms=%7.4f px  %s" % (rank, rms, path))
    if len(intrinsics_cases) >= 3:
        errs = [r for r, _ in intrinsics_cases]
        med = float(np.median(errs))
        worst = errs[0]
        print("    (median rms=%.4f px, worst=%.4f px)" % (med, worst))


def camera_calibrate(folder_path, checkerboard=CHECKERBOARD, square_size_m=square_size):
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError("Calibration folder does not exist: %s" % folder_path)

    images = get_images(folder_path)
    if not images:
        raise RuntimeError("No images found in %s (tried %s). Add calibration photos or fix the path." % (folder_path, ", ".join(_IMAGE_GLOBS)))

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    ok_paths = []
    gray_shape = None
    failed = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            failed.append((fname, "could not read image (missing, corrupt, or unsupported format)"))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            ok_paths.append(fname)
        else:
            failed.append((fname, "checkerboard not found (wrong size, blur, or no board)"))

    n_ok = len(objpoints)
    print("  Folder %s: %d / %d images with detected checkerboard" % (folder_path, n_ok, len(images)))
    if failed:
        print("  Failed images (%d):" % len(failed))
        for path, reason in failed:
            print("    - %s" % path)
            print("      %s" % reason)

    if n_ok == 0:
        raise RuntimeError("calibrateCamera needs at least one valid view; got 0 in %s. " "Checkerboard inner corners expected: %s (cols x rows of inner corners). " % (folder_path, checkerboard))
    if gray_shape is None:
        raise RuntimeError("Internal error: gray_shape unset despite n_ok > 0")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    print("  Overall calibrateCamera RMS reprojection: %.4f px" % ret)
    ranked = determine_error(objpoints, imgpoints, ok_paths, K, dist, rvecs, tvecs)
    print_report(ranked)

    if ret > 1.0:
        warnings.warn(f"Reprojection error is high ({ret:.4f} px)")

    return K, dist


def calibrate_cameras_and_save(calibration_folders, output_dir=".", K_list_filename="K_list.npy", dist_list_filename="dist_list.npy", checkerboard=CHECKERBOARD, square_size_m=square_size):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    K_list = []
    dist_list = []

    for i, folder in enumerate(calibration_folders):
        print(f"Calibrating camera {i} from folder: {folder}")
        K, dist, _ = camera_calibrate(folder, checkerboard=checkerboard, square_size_m=square_size_m)
        K_list.append(K)
        dist_list.append(dist)

    K_list_path = os.path.join(output_dir, K_list_filename)
    dist_list_path = os.path.join(output_dir, dist_list_filename)

    print("K_list:")
    for i, K in enumerate(K_list):
        print("  camera %d:\n%s" % (i, np.asarray(K)))
    print("dist_list:")
    for i, dist in enumerate(dist_list):
        print("  camera %d:\n%s" % (i, np.asarray(dist).ravel()))

    np.save(K_list_path, np.array(K_list, dtype=object))
    np.save(dist_list_path, np.array(dist_list, dtype=object))

    print(f"Saved {K_list_path} and {dist_list_path}")
    return K_list, dist_list


if __name__ == "__main__":
    calibration_folders = ["Video_Camera_Processing/intrinsics_calibration_camera_1", "Video_Camera_Processing/intrinsics_calibration_camera_0"]
    output_dir = "Video_Camera_Processing"
    calibrate_cameras_and_save(calibration_folders, output_dir=output_dir)
