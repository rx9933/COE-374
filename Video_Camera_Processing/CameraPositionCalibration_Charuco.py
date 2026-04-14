import cv2
import numpy as np


def create_board(squares_x=5, squares_y=7, square_length=0.04, marker_length=0.02):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    return board, aruco_dict

def detect_markers(image, board, aruco_dict):
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)
    if ids is None:
        return None, None
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board
    )

    if retval is None or retval < 4:
        print("No Charuco corners detected")
        return None, None
    return charuco_corners, charuco_ids


def board_to_world_points(board, charuco_ids, R_board, t_board):
    obj_pts_board = board.chessboardCorners[charuco_ids.flatten()]
    obj_pts_world = (R_board @ obj_pts_board.T + t_board).T
    return obj_pts_world


def estimate_camera_pos(image, boards, camera_matrix, dist_coeffs):
    all_obj_points = []
    all_img_points = []

    for board_data in boards:
        board = board_data["board"]
        aruco_dict = board_data["aruco_dict"]
        Rb = board_data["R"]
        tb = board_data["t"]

        charuco_corners, charuco_ids = detect_markers(image, board, aruco_dict)

        if charuco_corners is None:
            continue

        obj_pts_world = board_to_world_points(board, charuco_ids, Rb, tb)
        img_pts = charuco_corners.reshape(-1, 2)

        all_obj_points.append(obj_pts_world)
        all_img_points.append(img_pts)

    if len(all_obj_points) == 0:
        raise ValueError("No Charuco boards detected")

    all_obj_points = np.vstack(all_obj_points).astype(np.float32)
    all_img_points = np.vstack(all_img_points).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        all_obj_points,
        all_img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec

    return rvec, tvec, camera_position



def visualize_camera_layout(camera_position, rvec):
    import matplotlib.pyplot as plt
    pos = camera_position.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pos[0], pos[1], pos[2], c="blue", s=100)
    ax.text(pos[0], pos[1], pos[2], "Camera")

    R, _ = cv2.Rodrigues(rvec)
    direction = R.T @ np.array([0, 0, 1])

    ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], length=0.5, color='green')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def main():
    image = cv2.imread("Video_Camera_Processing/extrinsics_calibration/camera_0.png")
    if image is None:
        raise ValueError("Image not found")

    camera_matrix = np.load("Video_Camera_Processing/K_list.npy", allow_pickle=True)[0]
    dist_coeffs = np.load("Video_Camera_Processing/dist_list.npy", allow_pickle=True)[0]

    board, aruco_dict = create_board(squares_x=8, squares_y=9, square_length=0.04, marker_length=0.02)

    boards = [
        {
            "board": board,
            "aruco_dict": aruco_dict,
            "R": np.eye(3),
            "t": np.array([[0], [0], [0]])
        },
        {
            "board": board,
            "aruco_dict": aruco_dict,
            "R": np.eye(3),
            "t": np.array([[2], [0], [0]])
        }
    ]

    rvec, tvec, camera_position = estimate_camera_pos(image, boards, camera_matrix, dist_coeffs)

    print("\nCamera position:")
    print(camera_position.flatten())

    visualize_camera_layout(camera_position, rvec)


if __name__ == "__main__":
    main()