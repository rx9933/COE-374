import numpy as np
from dataclasses import dataclass
import time

def normalize(v):
    return np.asarray(v, dtype=float).reshape(-1) / np.linalg.norm(v)

def pixel_to_ray(K, u, v):
    p = np.array([u, v, 1.0], dtype=float)
    x = np.linalg.solve(K, p)
    return normalize(x)


def triangulate(r0, r1, R_01, t):
    a = normalize(r0)
    b = normalize(R_01 @ r1)
    lam, mu = np.linalg.solve(np.array([[a @ a, -a @ b], [a @ b, -b @ b]], dtype=float), np.array([a @ t, b @ t], dtype=float))
    P0 = lam * a
    P1 = t + mu * b
    P_hat = 0.5 * (P0 + P1)
    ray_gap = float(np.linalg.norm(P0 - P1))
    return P_hat, ray_gap


def re_length(P_tip_hat, P_tail_hat, L):
    c = 0.5 * (P_tip_hat + P_tail_hat)
    d = P_tip_hat - P_tail_hat
    d_hat = normalize(d)
    P_tip = c + 0.5 * L * d_hat
    P_tail = c - 0.5 * L * d_hat
    return P_tip, P_tail


def camera_to_absolute(X_c, R_w, t_w):
    return R_w @ X_c + t_w


def get_3d_pos(K, tip_uv_t0, tip_uv_t1, tail_uv_t0, tail_uv_t1, R_01, t_01, L, R_w=None, t_w=None):
    r_tip_0 = pixel_to_ray(K, *tip_uv_t0)
    r_tip_1 = pixel_to_ray(K, *tip_uv_t1)
    r_tail_0 = pixel_to_ray(K, *tail_uv_t0)
    r_tail_1 = pixel_to_ray(K, *tail_uv_t1)

    P_tip_hat, gap_tip = triangulate(r_tip_0, r_tip_1, R_01, t_01)
    P_tail_hat, gap_tail = triangulate(r_tail_0, r_tail_1, R_01, t_01)

    P_tip_c, P_tail_c = re_length(P_tip_hat, P_tail_hat, L)

    out = {}
    out["P_tip_cam"] = P_tip_c
    out["P_tail_cam"] = P_tail_c
    out["midpoint_cam"] = 0.5 * (P_tip_c + P_tail_c)
    out["direction_cam"] = normalize(P_tip_c - P_tail_c)
    out["ray_gap_tip"] = gap_tip
    out["ray_gap_tail"] = gap_tail
    out["P_tip_world"] = camera_to_absolute(P_tip_c, R_w, t_w)
    out["P_tail_world"] = camera_to_absolute(P_tail_c, R_w, t_w)
    out["midpoint_world"] = 0.5 * (out["P_tip_world"] + out["P_tail_world"])
    out["direction_world"] = normalize(out["P_tip_world"] - out["P_tail_world"])

    return out

#===============================================================================================================
#=================================================EKF===========================================================
#===============================================================================================================
@dataclass
class ProjectileEKF:    
    x: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    g: float = 9.81

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).reshape(6)
        self.P = np.asarray(self.P, dtype=float).reshape(6, 6)
        self.Q = np.asarray(self.Q, dtype=float).reshape(6, 6)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)

    def predict(self, dt):
        a = np.array([0.0, 0.0, -self.g], dtype=float)

        p = self.x[0:3]
        v = self.x[3:6]

        p_pred = p + v * dt + 0.5 * a * dt * dt
        v_pred = v + a * dt

        self.x[0:3] = p_pred
        self.x[3:6] = v_pred

        F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float).reshape(3)

        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        y = z - (H @ self.x)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P


def init_ekf(p0, v0, P0_diag, Q_diag, R_diag, g=9.81):  
    x0 = np.concatenate([p0, v0], axis=0)

    P0 = np.diag(P0_diag)
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    return ProjectileEKF(x=x0, P=P0, Q=Q, R=R, g=g)


#===============================================================================================================
#=============== Pos to Angle =======================================
#===============================================================================================================

def camera_yaw_pitch_to_predicted_target(drone_pos, target_pos_t, target_pos_tminus1, dt, n=1):
    v_t = (target_pos_t - target_pos_tminus1) / dt
    target_pos_t_pred = target_pos_t + v_t * (n * dt)

    rel = target_pos_t_pred - drone_pos
    dx, dy, dz = rel

    yaw = np.arctan2(dy, dx)
    pitch = np.arctan2(dz, np.sqrt(dx*dx + dy*dy))
    return yaw, pitch
    
#===============================================================================================================
#=================================================Main==========================================================
#===============================================================================================================


if __name__ == "__main__":

    
    K = np.array([[1200.0, 0.0, 640.0],
                  [0.0, 1200.0, 360.0],
                  [0.0, 0.0, 1.0]])

    tip_uv_t0 = np.array([700.0, 340.0])
    tip_uv_t1 = np.array([710.0, 342.0])
    tail_uv_t0 = np.array([580.0, 380.0])
    tail_uv_t1 = np.array([590.0, 382.0])

    R_01 = np.eye(3)
    t_01 = np.array([0.2, 0.0, 0.0])

    L = 2.6

    recon = get_3d_pos(K=K, tip_uv_t0=tip_uv_t0, tip_uv_t1=tip_uv_t1, tail_uv_t0=tail_uv_t0, tail_uv_t1=tail_uv_t1, R_01=R_01, t_01=t_01, L=L, R_w=None, t_w=None)
    print("Reconstructed tip (cam):", recon["P_tip_cam"])
    print("Reconstructed tail (cam):", recon["P_tail_cam"])
    print("Ray gaps (tip, tail):", recon["ray_gap_tip"], recon["ray_gap_tail"])

    ekf = init_ekf(p0=[0.0, 0.0, 1.5], v0=[20.0, 0.0, 8.0], P0_diag=[1.0, 1.0, 1.0, 10.0, 10.0, 10.0], Q_diag=[0.01, 0.01, 0.01, 0.1, 0.1, 0.1], R_diag=[0.05, 0.05, 0.08], g=9.81)

    dt = 1.0 / 30.0
    z_meas = np.array([0.7, 0.05, 1.55])

    ekf.predict(dt)
    ekf.update(z_meas)

    print("EKF state [p; v]:", ekf.x)
    print("EKF position:", ekf.x[:3], "EKF velocity:", ekf.x[3:])
    # drone code
    drone = np.array([0, 0, 10])
    x_tm1 = np.array([5, 2, 1])
    x_t   = np.array([5.5, 2.3, 1.2])
    yaw, pitch = camera_yaw_pitch_to_predicted_target(drone, x_t, x_tm1, dt=0.1, n_steps=3)
    print("yaw (rad):", yaw, "pitch (rad):", pitch)


