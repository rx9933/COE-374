import cv2
import numpy as np

def detect_markers(image):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if ids is None:
        print("No markers detected")
        return None, None

    print(f"Detected markers: {ids.flatten()}")
    return corners, ids.flatten()



def build_world_points(corners, ids, marker_length, marker_world_positions):
    object_points = []
    image_points = []

    half = marker_length / 2

    marker_corners = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ])

    for i, marker_id in enumerate(ids):

        if marker_id not in marker_world_positions:
            continue

        center = marker_world_positions[marker_id]
        world_corners = marker_corners + np.array(center)

        object_points.extend(world_corners)
        image_points.extend(corners[i][0])

    return np.array(object_points, dtype=np.float32), np.array(image_points, dtype=np.float32)


def estimate_camera_pose(object_points, image_points, camera_matrix, dist_coeffs):

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)

    camera_position = -R.T @ tvec

    return rvec, tvec, camera_position


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def get_camera_pose(image, marker_length, marker_world_positions, intrinsics, dist_coeffs):
    corners, ids = detect_markers(image)
    if corners is None:
        return None, None, None
    object_points, image_points = build_world_points(corners, ids, marker_length, marker_world_positions)
    rvec, tvec, camera_position = estimate_camera_pose(object_points, image_points, intrinsics, dist_coeffs)
    return rvec, tvec, camera_position

def main():

    image = cv2.imread("test_image.jpg")

    if image is None:
        raise ValueError("Failed to load image")

    corners, ids = detect_markers(image)

    if corners is None:
        raise ValueError("No markers detected")
    
    #define the world positions of the markers
    marker_world_positions = {
        0: (0.0, 0.0, 0.0),
        1: (2.0, 0.0, 0.0),
        2: (0.0, 2.0, 0.0),
        3: (2.0, 2.0, 0.0),
        4: (1.0, 3.0, 0.0)
    }

    marker_length = 0.20  # meters

    camera_matrix = np.load("Video_Camera_Processing/K_list.npy")
    dist_coeffs = np.load("Video_Camera_Processing/dist.npy")

    object_points, image_points = build_world_points(corners, ids, marker_length, marker_world_positions)
    rvec, tvec, camera_position = estimate_camera_pose(object_points, image_points, camera_matrix, dist_coeffs)

    print("\nCamera rotation vector:")
    print(rvec)

    print("\nCamera translation vector:")
    print(tvec)

    print("\nCamera position in world coordinates:")
    print(camera_position.flatten())

    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.5)

    return image


if __name__ == "__main__":
    main()