import cv2
import os
import numpy as np
"""
(0, 0, 0) -> x
  |
 \ / y
"""

def gen_boards(output_dir="Video_Camera_Processing/markers", num_boards=2, squares_x=8, squares_y=9, square_length=0.04, marker_length=0.02, pixels_per_square=200):
    os.makedirs(output_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)

    markers_per_board = squares_x * squares_y

    for b in range(num_boards):
        id_offset = b * markers_per_board
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

        width = squares_x * pixels_per_square
        height = squares_y * pixels_per_square
        img = board.generateImage((width, height))
        filename = os.path.join(output_dir, f"charuco_board_{b}.png")
        cv2.imwrite(filename, img)

        print(f"Saved: {filename}")


def gen_aruco_grids(output_dir="Video_Camera_Processing/markers", num_boards=2, markers_x=4, markers_y=5, marker_length_m=0.04, marker_separation_m=0.02, width_px=1600, dictionary_id=cv2.aruco.DICT_4X4_1000):
    os.makedirs(output_dir, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    n = markers_x * markers_y
    aspect = markers_x / markers_y
    height_px = int(round(width_px / aspect))

    for b in range(num_boards):
        first_id = b * n
        ids = np.arange(first_id, first_id + n, dtype=np.int32)
        board = cv2.aruco.GridBoard(
            (markers_x, markers_y),
            marker_length_m,
            marker_separation_m,
            aruco_dict,
            ids,
        )
        img = board.generateImage((width_px, height_px))
        filename = os.path.join(output_dir, f"aruco_grid_board_{b}.png")
        cv2.imwrite(filename, img)
        print(f"Saved: {filename} (marker IDs {first_id}..{first_id + n - 1})")


if __name__ == "__main__":
    gen_boards()