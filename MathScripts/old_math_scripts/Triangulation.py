"""
Method 1 (Fixed Camera Parameters):
Assume all camera intrinsics and poses are known exactly, 
then estimate the 3D object position by triangulating from multiple pixel observations and refining it with nonlinear least squares.
Ground Control Points (GCPs) can optionally act as additional constraints to improve robustness.
"""


import numpy as np
from scipy.optimize import least_squares
import time


def project_point(P, X):
    X_h = np.hstack([X, 1.0])
    x = P @ X_h
    return x[:2] / x[2]


def dlt_triangulation(P_list, pixels):
    A = []

    for P, (u, v) in zip(P_list, pixels):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])

    A = np.vstack(A)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]

    return X


def reprojection_residual(X, P_list, pixels, weights=None, ground_points=None, ground_weight=0.0):
    residuals = []
    for i, (P, obs) in enumerate(zip(P_list, pixels)):
        pred = project_point(P, X)
        r = pred - obs
        if weights is not None:
            r = r * weights[i]
        residuals.append(r)

    if ground_points is not None and len(ground_points) > 0:
        for G in ground_points:
            r = np.sqrt(ground_weight) * (X - G)
            residuals.append(r)

    return np.concatenate(residuals)

def triangulate_nonlinear(P_list, pixels, pixel_sigma=1.0, weights=None, ground_points=None, ground_weight=1.0):
    X0 = dlt_triangulation(P_list, pixels)

    result = least_squares(reprojection_residual, X0, args=(P_list, pixels, weights, ground_points, ground_weight), method="lm")
    X_opt = result.x
    J = result.jac
    JTJ = J.T @ J

    try:
        cov = pixel_sigma**2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        print("Singular matrix, cannot compute covariance")
        cov = np.full((3, 3), np.nan)

    return X_opt, cov, result


#==============================================
# Example usage
#==============================================


def camera_look_at(cam_position, target, up=np.array([0, 0, 1])):
    z = target - cam_position
    z = z / (np.linalg.norm(z) + 1e-10)
    up = np.asarray(up, dtype=float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-10:
        x = np.array([1, 0, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    t = -R @ cam_position
    return R, t


def build_P(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])



if __name__ == "__main__":
    np.random.seed(42)
    n_cameras = 3
    pixel_noise_std = 2.0

    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    center = np.array([0.0, 0.0, 0.0])
    radius = 4.0
    height = 2.0
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    cam_positions = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(3, height),
    ])

    P_list = []
    for i in range(n_cameras):
        R, t = camera_look_at(cam_positions[i], center)
        P_list.append(build_P(K, R, t))

    X_obj_true = np.array([0.0, 0.0, 0.5])
    ground_points = [
        np.array([-0.4, 0.3, 0.0]),
        np.array([0.4, -0.3, 0.0]),
        np.array([0.0, 0.4, 0.0]),
    ]

    pixels = []
    for i in range(n_cameras):
        uv = project_point(P_list[i], X_obj_true)
        uv += np.random.randn(2) * pixel_noise_std
        pixels.append(uv)

    start_time = time.time()
    X, cov, result = triangulate_nonlinear(
        P_list, pixels,
        pixel_sigma=pixel_noise_std,
        ground_points=ground_points,
        ground_weight=10.0
    )
    end_time = time.time()

    print("Time taken: {:.4f} s".format(end_time - start_time))
    print("True object position:", X_obj_true)
    print("Estimated 3D point: ", X)
    print("Object error (m):   ", np.linalg.norm(X - X_obj_true))
    print("Covariance:\n", cov)