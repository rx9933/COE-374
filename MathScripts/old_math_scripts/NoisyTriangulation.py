"""
Method 2 (Joint Camera and Object Optimization with GCPs):
Assume camera poses may be inaccurate and jointly optimize both the 3D object position and the camera parameters 
using pixel observations and precise GCP measurements. 
The GCPs anchor the solution in absolute space while bundle adjustment refines everything simultaneously.
"""

import numpy as np
from scipy.optimize import least_squares
import time


def rodrigues_to_matrix(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        print("Small rotation, using identity matrix")  
        return np.eye(3)
    else:
        k = rvec / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]) #rotation matrix
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K) #Rodrigues formula
        return R

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


def bundle_residual(params, K_list, pixel_obj_list, local_gcps, absolute_gcps, n_cameras, m_gcps, weights=None):
    X_obj = params[:3]
    rvecs = params[3:3 + 3 * n_cameras].reshape((n_cameras, 3))
    tvecs = params[3 + 3 * n_cameras:].reshape((n_cameras, 3))

    residuals = []

    for i in range(n_cameras):
        R = rodrigues_to_matrix(rvecs[i])
        t = tvecs[i]
        P = K_list[i] @ np.hstack([R, t.reshape(3, 1)])

        pred = project_point(P, X_obj)
        r = pred - pixel_obj_list[i]
        if weights is not None:
            r = r * weights[i]
        residuals.append(r)

        for j in range(m_gcps):
            gcp_pred = project_point(P, absolute_gcps[j])
            r_gcp = local_gcps[i][j] - gcp_pred
            if weights is not None:
                r_gcp = r_gcp * weights[n_cameras + j]
            residuals.append(r_gcp)

    return np.concatenate(residuals)


def triangulate_nonlinear(K_list, rvecs_init, tvecs_init, pixels, local_gcps=None, absolute_gcps=None, pixel_sigma=1.0, weights=None):

    n_cameras = len(K_list)
    m_gcps = 0 if absolute_gcps is None else len(absolute_gcps)
    if local_gcps is None:
        local_gcps = [[]] * n_cameras
    if absolute_gcps is None:
        absolute_gcps = []


    P_list = [K_list[i] @ np.hstack([rodrigues_to_matrix(rvecs_init[i]), tvecs_init[i].reshape(3, 1)]) for i in range(n_cameras)]
    X0 = dlt_triangulation(P_list, pixels)
    params_init = np.concatenate([X0, rvecs_init.ravel(), tvecs_init.ravel()])

    result = least_squares(bundle_residual, params_init, args=(K_list, pixels, local_gcps, absolute_gcps, n_cameras, m_gcps, weights), method="lm")

    X_opt = result.x[:3]
    rvecs_opt = result.x[3:3 + 3 * n_cameras].reshape((n_cameras, 3))
    tvecs_opt = result.x[3 + 3 * n_cameras:].reshape((n_cameras, 3))
    J = result.jac
    JTJ = J.T @ J

    # compute covariance
    try:
        cov_full = pixel_sigma**2 * np.linalg.inv(JTJ)
        cov = cov_full[:3, :3]
    except np.linalg.LinAlgError:
        print("Singular matrix, cannot compute covariance")
        cov = np.full((3, 3), np.nan)

    return X_opt, rvecs_opt, tvecs_opt, cov, result


#==============================================
# Example usage
#==============================================


def matrix_to_rodrigues(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3)
    sin_theta = np.sin(theta)
    return (theta / (2 * sin_theta)) * np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]
    ])


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


if __name__ == "__main__":
    np.random.seed(42)
    n_cameras = 3
    pixel_noise_std = 2.0
    cam_pose_noise = 0.03

    # Camera intrinsics (same for all)
    fx, fy, cx, cy = 800, 800, 320, 240
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_list = [K.copy() for _ in range(n_cameras)]

    # Triangle: cameras at vertices, distance 4 from center, height 2, facing center (0,0,0)
    center = np.array([0.0, 0.0, 0.0])
    radius = 4.0
    height = 2.0
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    cam_positions = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(3, height),
    ])

    R_true_list = []
    t_true_list = []
    for i in range(n_cameras):
        R, t = camera_look_at(cam_positions[i], center)
        R_true_list.append(R)
        t_true_list.append(t)
    rvecs_true = np.array([matrix_to_rodrigues(R) for R in R_true_list])
    tvecs_true = np.array(t_true_list)

    # Slightly wrong initial poses (noisy camera positioning)
    rvecs_init = rvecs_true + np.random.randn(n_cameras, 3) * cam_pose_noise
    tvecs_init = tvecs_true + np.random.randn(n_cameras, 3) * cam_pose_noise

    # Object point and GCPs inside the triangle (in front of all cameras)
    X_obj_true = np.array([0.0, 0.0, 0.5])
    absolute_gcps = [
        np.array([-0.4, 0.3, 0.0]),
        np.array([0.4, -0.3, 0.0]),
        np.array([0.0, 0.4, 0.0]),
    ]
    m_gcps = len(absolute_gcps)

    # Build true projection matrices and generate noisy pixel observations
    P_true_list = [K_list[i] @ np.hstack([R_true_list[i], t_true_list[i].reshape(3, 1)]) for i in range(n_cameras)]

    pixels = []
    for i in range(n_cameras):
        uv = project_point(P_true_list[i], X_obj_true)
        uv += np.random.randn(2) * pixel_noise_std
        pixels.append(uv)

    local_gcps = []
    for i in range(n_cameras):
        gcp_pixels = []
        for gcp in absolute_gcps:
            uv = project_point(P_true_list[i], gcp)
            uv += np.random.randn(2) * pixel_noise_std
            gcp_pixels.append(uv)
        local_gcps.append(gcp_pixels)

    start_time = time.time()
    X, rvecs_opt, tvecs_opt, cov, result = triangulate_nonlinear(
        K_list, rvecs_init, tvecs_init, pixels,
        local_gcps=local_gcps, absolute_gcps=absolute_gcps,
        pixel_sigma=pixel_noise_std
    )
    end_time = time.time()

    print("Time taken: {:.4f} s".format(end_time - start_time))
    print("True object position:", X_obj_true)
    print("Estimated 3D point: ", X)
    print("Object error (m):    ", np.linalg.norm(X - X_obj_true))
    print("Refined camera rotations (axis-angle):\n", rvecs_opt)
    print("Refined camera translations:\n", tvecs_opt)
    print("Covariance (object position):\n", cov)