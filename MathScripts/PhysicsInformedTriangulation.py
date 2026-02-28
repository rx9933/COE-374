"""
Method 3 (Physics-Informed Triangulation):
Assume camera poses may be inaccurate and jointly optimize both the 3D object position and the camera parameters 
using pixel observations and precise GCP measurements. 
The GCPs anchor the solution in absolute space while bundle adjustment refines everything simultaneously.
The physics constraint is that the object is moving in a parabolic trajectory.
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


def bundle_residual(params, K_list, pixels, local_gcps, absolute_gcps, n_cameras, n_timesteps, omega_gcp=1.0, omega_phys=1.0, dt=1.0, g=None, drag=0.0):
    X_vars = params[:3 * n_timesteps].reshape((n_timesteps, 3))
    rvecs = params[3 * n_timesteps:3 * n_timesteps + 3 * n_cameras].reshape((n_cameras, 3))
    tvecs = params[3 * n_timesteps + 3 * n_cameras:].reshape((n_cameras, 3))

    residuals = []
    P_list = [K_list[i] @ np.hstack([rodrigues_to_matrix(rvecs[i]), tvecs[i].reshape(3, 1)]) for i in range(n_cameras)]

    for t in range(n_timesteps):
        X_t = X_vars[t]
        for i in range(n_cameras):
            pred = project_point(P_list[i], X_t)
            r = pred - pixels[i][t]
            residuals.append(r)

    if absolute_gcps is not None and len(absolute_gcps) > 0:
        for i in range(n_cameras):
            for j, G in enumerate(absolute_gcps):
                if local_gcps[i][j] is not None:
                    pred = project_point(P_list[i], G)
                    r = omega_gcp * (pred - local_gcps[i][j])
                    residuals.append(r)

    for t in range(1, n_timesteps - 1):
        X_prev = X_vars[t - 1]
        X_curr = X_vars[t]
        X_next = X_vars[t + 1]
        phys_res = X_next - 2 * X_curr + X_prev - g * dt**2 - drag * dt**2
        residuals.append(omega_phys * phys_res)

    return np.concatenate(residuals)


def optimize_trajectory(K_list, pixels, local_gcps=None, absolute_gcps=None, dt=1.0, g=None, drag=0.0, rvecs_init=None, tvecs_init=None, pixel_sigma=1.0):
    if g is None:
        g = np.array([0, -9.81, 0])
    n_cameras = len(K_list)
    n_timesteps = len(pixels[0])
    if local_gcps is None:
        local_gcps = []
    if absolute_gcps is None:
        absolute_gcps = []
    m_gcps = len(absolute_gcps)
    if m_gcps > 0 and (len(local_gcps) == 0 or len(local_gcps[0]) != m_gcps):
        local_gcps = [[None] * m_gcps for _ in range(n_cameras)]

    if rvecs_init is None:
        rvecs_init = np.zeros((n_cameras, 3))
    if tvecs_init is None:
        tvecs_init = np.zeros((n_cameras, 3))

    X_init = []
    for t in range(n_timesteps):
        P_list_tmp = [K_list[i] @ np.hstack([rodrigues_to_matrix(rvecs_init[i]), tvecs_init[i].reshape(3, 1)]) for i in range(n_cameras)]
        pixel_t = [pixels[i][t] for i in range(n_cameras)]
        X_init.append(dlt_triangulation(P_list_tmp, pixel_t))
    X_init = np.array(X_init).flatten()

    params_init = np.concatenate([X_init, rvecs_init.ravel(), tvecs_init.ravel()])

    result = least_squares(bundle_residual, params_init, args=(K_list, pixels, local_gcps, absolute_gcps, n_cameras, n_timesteps, 1.0, 1.0, dt, g, drag), method="lm")

    X_opt = result.x[:3 * n_timesteps].reshape((n_timesteps, 3))
    rvecs_opt = result.x[3 * n_timesteps:3 * n_timesteps + 3 * n_cameras].reshape((n_cameras, 3))
    tvecs_opt = result.x[3 * n_timesteps + 3 * n_cameras:].reshape((n_cameras, 3))

    J = result.jac
    JTJ = J.T @ J
    try:
        cov_full = pixel_sigma**2 * np.linalg.inv(JTJ)
        cov = cov_full[:3 * n_timesteps, :3 * n_timesteps]
    except np.linalg.LinAlgError:
        cov = np.full((3 * n_timesteps, 3 * n_timesteps), np.nan)

    return X_opt, rvecs_opt, tvecs_opt, cov, result


#==============================================
# Example usage
#==============================================

def _matrix_to_rodrigues(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3)
    return (theta / (2 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _camera_look_at(cam_position, target, up=np.array([0, 0, 1])):
    z = (target - cam_position) / (np.linalg.norm(target - cam_position) + 1e-10)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-10:
        x = np.array([1, 0, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    t = -R @ cam_position
    return R, t


if __name__ == "__main__":
    print("Starting Physics-Informed Triangulation")
    np.random.seed(42)
    n_cameras = 3
    n_timesteps = 25
    dt = 0.04
    g = np.array([0.0, 0.0, -9.81])
    drag_coef = 0.2

    K_list = [np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]) for _ in range(n_cameras)]

    center = np.array([0.0, 0.0, 0.0])
    radius, height = 4.0, 2.0
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    cam_positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(3, height)])
    R_list = []
    t_list = []
    for i in range(n_cameras):
        R, t = _camera_look_at(cam_positions[i], center)
        R_list.append(R)
        t_list.append(t)
    rvecs_init = np.array([_matrix_to_rodrigues(R) for R in R_list])
    tvecs_init = np.array(t_list)

    X0 = np.array([0.0, 0.0, 1.5])
    V0 = np.array([1.0, 0.0, 2.5])
    traj_true = []
    x, v = X0.copy(), V0.copy()
    for t in range(n_timesteps):
        traj_true.append(x.copy())
        acc = g - drag_coef * v
        v = v + acc * dt
        x = x + v * dt
        if x[2] < 0.2:
            break
    n_timesteps = len(traj_true)
    traj_true = np.array(traj_true)

    pixels = []
    for i in range(n_cameras):
        P = K_list[i] @ np.hstack([R_list[i], t_list[i].reshape(3, 1)])
        pixels.append([project_point(P, traj_true[t]) + np.random.randn(2) * 1.0 for t in range(n_timesteps)])

    absolute_gcps = [np.array([-0.3, 0.2, 0.0]), np.array([0.3, -0.2, 0.0])]
    local_gcps = []
    for i in range(n_cameras):
        P = K_list[i] @ np.hstack([R_list[i], t_list[i].reshape(3, 1)])
        local_gcps.append([project_point(P, absolute_gcps[j]) + np.random.randn(2) * 1.0 for j in range(len(absolute_gcps))])
    time_start = time.time()
    X_opt, rvecs_opt, tvecs_opt, cov, result = optimize_trajectory(
        K_list, pixels, local_gcps, absolute_gcps, dt=dt, g=g, drag=0.0,
        rvecs_init=rvecs_init, tvecs_init=tvecs_init, pixel_sigma=1.0
    )
    time_end = time.time()
    print("Time taken: {:.4f} s".format(time_end - time_start))
    print("Trajectory shape: {} timesteps x 3".format(n_timesteps))
    print("Optimized (first 3, last 2):\n", np.vstack([X_opt[:3], X_opt[-2:]]))
    print("True (first 3, last 2):\n", np.vstack([traj_true[:3], traj_true[-2:]]))
    print("Mean position error (m):", np.mean(np.linalg.norm(X_opt - traj_true, axis=1)))
    print("Optimized camera rotations (rvecs):\n", rvecs_opt)
    print("Optimized camera translations:\n", tvecs_opt)
    print("Covariance (trajectory positions, 3x3 per timestep on diagonal):")
    #for t in range(n_timesteps):
    #    cov_t = cov[3 * t:3 * t + 3, 3 * t:3 * t + 3]
    #    print("  t={}: std (x,y,z) ~ {}".format(t, np.sqrt(np.diag(cov_t))))

    # Plot: true vs optimized trajectory and uncertainty ellipsoids
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        cov_scale = 1.0
        if np.all(np.isfinite(cov)):
            max_std = np.max(np.sqrt(np.diag(cov)))
            if max_std > 1.0:
                cov_scale = 0.5 / max_std

        def _ellipsoid_points(center, cov_3x3, n_sigma=1.0, n_pts=32, scale=1.0):
            if np.any(np.isnan(cov_3x3)) or np.any(np.linalg.eigvalsh(cov_3x3) <= 0):
                return None, None, None
            eigs, Q = np.linalg.eigh(cov_3x3)
            eigs = np.maximum(eigs, 1e-12)
            radii = scale * n_sigma * np.sqrt(eigs)
            u = np.linspace(0, 2 * np.pi, n_pts)
            v = np.linspace(0, np.pi, n_pts)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            shape = x.shape
            pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) @ (Q * radii).T
            pts += center
            return pts[:, 0].reshape(shape), pts[:, 1].reshape(shape), pts[:, 2].reshape(shape)

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], "b-o", label="True", markersize=8)
        ax1.plot(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], "r-s", label="Optimized", markersize=6)
        for t in range(n_timesteps):
            cov_t = cov[3 * t:3 * t + 3, 3 * t:3 * t + 3]
            xe, ye, ze = _ellipsoid_points(X_opt[t], cov_t, n_sigma=1.0, scale=cov_scale)
            if xe is not None:
                ax1.plot_surface(xe, ye, ze, alpha=0.15, color="red")
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
        ax1.legend(); ax1.set_title("3D: true vs optimized + 1σ ellipsoids")

        ax2 = fig.add_subplot(132)
        ax2.plot(traj_true[:, 0], traj_true[:, 1], "b-o", label="True")
        ax2.plot(X_opt[:, 0], X_opt[:, 1], "r-s", label="Optimized")
        for t in range(n_timesteps):
            cov_t = cov[3 * t:3 * t + 3, 3 * t:3 * t + 3]
            if np.any(np.isnan(cov_t)):
                continue
            eigs, Q = np.linalg.eigh(cov_t[:2, :2])
            eigs = np.maximum(eigs, 1e-12)
            angles = np.linspace(0, 2 * np.pi, 50)
            circle = np.column_stack([np.cos(angles), np.sin(angles)])
            ellipse = X_opt[t, :2] + cov_scale * (circle @ (Q * np.sqrt(eigs)).T)
            ax2.plot(ellipse[:, 0], ellipse[:, 1], "r-", alpha=0.5)
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.legend(); ax2.set_title("xy + 1σ ellipses")
        ax2.set_aspect("equal"); ax2.grid(True)

        ax3 = fig.add_subplot(133)
        ax3.plot(traj_true[:, 0], traj_true[:, 2], "b-o", label="True")
        ax3.plot(X_opt[:, 0], X_opt[:, 2], "r-s", label="Optimized")
        for t in range(n_timesteps):
            cov_t = cov[3 * t:3 * t + 3, 3 * t:3 * t + 3]
            if np.any(np.isnan(cov_t)):
                continue
            c = np.array([[cov_t[0, 0], cov_t[0, 2]], [cov_t[2, 0], cov_t[2, 2]]])
            eigs, Q = np.linalg.eigh(c)
            eigs = np.maximum(eigs, 1e-12)
            angles = np.linspace(0, 2 * np.pi, 50)
            circle = np.column_stack([np.cos(angles), np.sin(angles)])
            ellipse = X_opt[t, [0, 2]] + cov_scale * (circle @ (Q * np.sqrt(eigs)).T)
            ax3.plot(ellipse[:, 0], ellipse[:, 1], "r-", alpha=0.5)
        ax3.set_xlabel("x"); ax3.set_ylabel("z"); ax3.legend(); ax3.set_title("xz + 1σ ellipses")
        ax3.set_aspect("equal"); ax3.grid(True)

        plt.tight_layout()
        plt.savefig("trajectory_and_uncertainty.png", dpi=120)
        plt.show()
    except ImportError:
        print("Install matplotlib to plot trajectory and covariances.")