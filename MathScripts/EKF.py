import numpy as np
from dataclasses import dataclass
import time

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