import os

import numpy as np

OUTPUT_DIR = "Video_Camera_Processing"
K_LIST_FILENAME = "K_list.npy"
DIST_LIST_FILENAME = "dist_list.npy"

intrinsics_list = [
    np.array(
        [[3480.0, 0.0, 960.0], [0.0, 3480.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    np.array(
        [[3480.0, 0.0, 960.0], [0.0, 3480.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
]

dist_coeffs_list = [
    np.zeros((5, 1), dtype=np.float64),
    np.zeros((5, 1), dtype=np.float64),
]


def save_approximate_intrinsics(
    output_dir=None,
    k_filename=None,
    dist_filename=None,
    k_list=None,
    dist_list=None,
):
    output_dir = os.path.abspath(output_dir or OUTPUT_DIR)
    k_filename = k_filename or K_LIST_FILENAME
    dist_filename = dist_filename or DIST_LIST_FILENAME
    k_list = k_list if k_list is not None else intrinsics_list
    dist_list = dist_list if dist_list is not None else dist_coeffs_list

    os.makedirs(output_dir, exist_ok=True)
    k_path = os.path.join(output_dir, k_filename)
    dist_path = os.path.join(output_dir, dist_filename)

    np.save(k_path, np.array(k_list, dtype=object))
    np.save(dist_path, np.array(dist_list, dtype=object))
    print(f"Saved {k_path}")
    print(f"Saved {dist_path}")
    return k_path, dist_path


if __name__ == "__main__":
    save_approximate_intrinsics()
