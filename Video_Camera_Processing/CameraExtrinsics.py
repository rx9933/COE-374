import cv2
import numpy as np
from pathlib import Path

CAMERA_ID = 0

IMG_PATH = Path(__file__).resolve().parent / "extrinsics_calibration" / "extrinsics_0.png"
K_LIST_PATH = Path(__file__).resolve().parent / "K_list.npy"
DIST_LIST_PATH = Path(__file__).resolve().parent / "dist_list.npy"

DICT_ID = cv2.aruco.DICT_4X4_50
board_size = (4, 3)
sq_length = 0.06
marker_len = 0.045
USE_CROPPING = False

board_pos = [
    np.array([2.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 2.0, 0.0], dtype=np.float64),
    np.array([2.0, 2.0, 1.0], dtype=np.float64),
]

board_uv_crop = [
    (2505, 2332, 2848, 2632),
    (668, 2199, 866, 2442),
    (2491, 1462, 2666, 1631),
]
board_marker_id_ranges = [
    (0, 5),
    (6, 11),
    (12, 17),
]


def make_charuco_board(board_size, sq_length, marker_len, dictionary, id_min, id_max):
    ids = np.arange(id_min, id_max + 1, dtype=np.int32)
    return cv2.aruco.CharucoBoard(board_size, sq_length, marker_len, dictionary, ids)

def main():
    K_list = np.load(str(K_LIST_PATH), allow_pickle=True)
    dist_list = np.load(str(DIST_LIST_PATH), allow_pickle=True)

    K_full = np.asarray(K_list[CAMERA_ID], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist_list[CAMERA_ID], dtype=np.float64)

    image = cv2.imread(str(IMG_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMG_PATH}")
    full_h, full_w = image.shape[:2]

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)

    object_points = []
    image_points = []
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for board_id in range(len(board_pos)):

        world_pos = board_pos[board_id]
        if USE_CROPPING:
            x0, y0, x1, y1 = board_uv_crop[board_id]
            crop = image[y0:y1, x0:x1]
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            K_img = K_full.copy()
            K_img[0, 2] -= x0
            K_img[1, 2] -= y0
            origin_offset = np.array([float(x0), float(y0)], dtype=np.float64)
        else:
            img = gray_full
            K_img = K_full
            origin_offset = np.array([0.0, 0.0], dtype=np.float64)

        corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
        if ids is None:
            print(f"[warn] No markers for board {board_id}")
            continue

        id_min, id_max = board_marker_id_ranges[board_id]
        ids_flat = ids.flatten()
        keep_mask = (ids_flat >= id_min) & (ids_flat <= id_max)
        filtered_count = int(np.sum(keep_mask))

        if filtered_count == 0:
            print(f"[warn] Board {board_id}: no IDs in expected range [{id_min}, {id_max}], detected={ids_flat.tolist()}\n")
            continue

        corners = [c for c, keep in zip(corners, keep_mask) if keep]
        ids = ids[keep_mask].reshape(-1, 1)
        print(f"Board {board_id}: detected={ids_flat.tolist()} ")

        board = make_charuco_board(board_size, sq_length, marker_len, dictionary, id_min, id_max)
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board, cameraMatrix=K_img, distCoeffs=dist)

        print(f"Board {board_id}: charuco_corners={charuco_corners}")

        if charuco_corners is None or len(charuco_corners) < 4:
            print(f"[warn] Not enough Charuco corners for board {board_id}\n")
            continue

        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, K_img, dist, None, None)

        if not success:
            print(f"[warn] Pose failed for board {board_id}\n")
            continue

        gcp_origin = np.array([[0, 0, 0]], dtype=np.float64)
        origin_crop, _ = cv2.projectPoints(gcp_origin, rvec, tvec, K_img, dist)
        origin_crop = origin_crop.reshape(2)
        origin_full = origin_crop + origin_offset

        object_points.append(world_pos)
        image_points.append(origin_full)

        print(f"[info] Board {board_id}: origin_crop={origin_crop}, origin_full={origin_full}\n")

    if len(object_points) < 3:
        raise ValueError("Need at least 3 boards for PnP.\n")

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K_full,
        dist,
        flags=cv2.SOLVEPNP_AP3P
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    R_wc, _ = cv2.Rodrigues(rvec)

    print("\n=== CAMERA POS ===")
    print("Rotation matrix:")
    print(R_wc)
    print("\nCamera position:")
    print(tvec.ravel())


if __name__ == "__main__":
    main()