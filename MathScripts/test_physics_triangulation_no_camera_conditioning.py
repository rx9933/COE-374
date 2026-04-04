from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from Physics_Triangulation_No_Camera_Conditioning import (
    _camera_look_at,
    dlt_triangulation,
    optimize_trajectory,
    project_point,
    trajectory_residual,
)


def two_cameras():
    P0 = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    P1 = np.hstack([np.eye(3), np.array([[-0.55], [0.0], [0.0]])])
    return [P0, P1]


def main_style_scene(rng, *, n_cameras = 3, n_steps_cap = 25, dt = 0.04, g = None, drag_coef = 0.2, pixel_noise_sigma = 2.0, X0 = None, V0 = None, center = None, radius = 4.0, height = 2.0, z_floor = 0.2):
    if g is None:
        g = np.array([0.0, 0.0, -9.81])
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    if X0 is None:
        X0 = np.array([0.0, 0.0, 1.5])
    if V0 is None:
        V0 = np.array([1.0, 1.0, 2.5])

    K_list = [
        np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        for _ in range(n_cameras)
    ]
    if n_cameras == 3:
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    else:
        angles = 2 * np.pi * np.arange(n_cameras, dtype=np.float64) / n_cameras
    cam_positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_cameras, height)])
    R_list = []
    t_list = []
    for i in range(n_cameras):
        R, t = _camera_look_at(cam_positions[i], center)
        R_list.append(R)
        t_list.append(t)
    P_list = [
        K_list[i] @ np.hstack([R_list[i], t_list[i].reshape(3, 1)])
        for i in range(n_cameras)
    ]

    traj_true = []
    x, v = X0.copy(), V0.copy()
    for _ in range(n_steps_cap):
        traj_true.append(x.copy())
        acc = g - drag_coef * v
        v = v + acc * dt
        x = x + v * dt
        if x[2] < z_floor:
            break
    traj_true = np.asarray(traj_true, dtype=np.float64)
    n_timesteps = traj_true.shape[0]

    pixels = []
    for i in range(n_cameras):
        pixels.append([project_point(P_list[i], traj_true[t])+ rng.normal(0.0, pixel_noise_sigma, size=2) for t in range(n_timesteps)])

    return {"P_list": P_list, "pixels": pixels, "traj_true": traj_true, "g": g, "dt": dt, "drag_coef": drag_coef, "n_timesteps": n_timesteps, "n_cameras": n_cameras}


class TestProjectPoint(unittest.TestCase):
    def test_simple_projection(self):
        P = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        X = np.array([2.0, 4.0, 2.0])
        uv = project_point(P, X)
        np.testing.assert_allclose(uv, [1.0, 2.0], rtol=0, atol=1e-12)

    def test_origin_on_optical_axis(self):
        P = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        X = np.array([0.0, 0.0, 1.0])
        uv = project_point(P, X)
        np.testing.assert_allclose(uv, [0.0, 0.0], rtol=0, atol=1e-12)

    def test_with_intrinsics(self):
        fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        R = np.eye(3)
        t = np.zeros(3)
        P = K @ np.hstack([R, t.reshape(3, 1)])
        X = np.array([1.0, 0.5, 5.0])
        uv = project_point(P, X)
        expected = np.array([fx * (1.0 / 5.0) + cx, fy * (0.5 / 5.0) + cy])
        np.testing.assert_allclose(uv, expected, rtol=0, atol=1e-9)


class TestDltTriangulation(unittest.TestCase):
    def test_recover_point_two_cameras(self):
        P0 = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        R1 = np.eye(3)
        t1 = np.array([-0.5, 0.0, 0.0])
        P1 = np.hstack([R1, t1.reshape(3, 1)])
        X_true = np.array([0.3, -0.2, 2.5])
        p0 = project_point(P0, X_true)
        p1 = project_point(P1, X_true)
        X_hat = dlt_triangulation([P0, P1], [p0, p1])
        np.testing.assert_allclose(X_hat, X_true, rtol=0, atol=1e-5)

    def test_recover_point_three_cameras(self):
        rng = np.random.default_rng(0)
        P_list = []
        for _ in range(3):
            R = rng.standard_normal((3, 3))
            q, _ = np.linalg.qr(R)
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            t = rng.standard_normal(3) * 0.3
            K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
            P_list.append(K @ np.hstack([q, t.reshape(3, 1)]))
        X_true = np.array([0.1, 0.15, 3.0])
        pixels = [project_point(P, X_true) for P in P_list]
        X_hat = dlt_triangulation(P_list, pixels)
        np.testing.assert_allclose(X_hat, X_true, rtol=0, atol=1e-4)


def drag_calc(X_prev, X_curr, g, drag, dt):
    return (2.0 * X_curr - X_prev + g * dt**2 + drag * dt / 2.0 * X_prev) / (1.0 + drag * dt / 2.0)

    
class TestTrajectoryResidual(unittest.TestCase):
    def test_zero_variance(self):
        n_timesteps = 3
        n_cams = 2
        dt = 0.1
        g = np.array([0.0, 0.0, -10.0])
        drag = 0.25
        P_list = two_cameras()
        base = np.array([0.1, -0.05, 14.0])
        X0 = base.copy()
        X1 = base.copy()
        X2 = drag_calc(X0, X1, g, drag, dt)
        X_vars = np.stack([X0, X1, X2], axis=0)

        pixels = []
        for i in range(n_cams):
            row = [project_point(P_list[i], X_vars[t]) for t in range(n_timesteps)]
            pixels.append(row)

        params = np.concatenate([X_vars.ravel(), [drag]])
        res = trajectory_residual(
            params,
            P_list,
            pixels,
            n_timesteps,
            omega_phys=1.0,
            dt=dt,
            g=g,
            pixel_sigma=1.0,
            physics_sigma=1.0,
        )
        np.testing.assert_allclose(res, 0.0, atol=1e-9)

    def test_wrong_drag(self):
        n_timesteps = 3
        dt = 0.1
        g = np.array([0.0, 0.0, -9.81])
        drag_true = 0.1
        P_list = [
            np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
        ]
        base = np.array([0.0, 0.0, 8.0])
        X0 = base.copy()
        X1 = base.copy()
        X2 = drag_calc(X0, X1, g, drag_true, dt)
        X_vars = np.stack([X0, X1, X2], axis=0)
        pixels = [[project_point(P_list[0], X_vars[t]) for t in range(n_timesteps)]]

        params_bad_drag = np.concatenate([X_vars.ravel(), [99.0]])
        res = trajectory_residual(
            params_bad_drag,
            P_list,
            pixels,
            n_timesteps,
            dt=dt,
            g=g,
            pixel_sigma=1.0,
            physics_sigma=1.0,
        )
        self.assertGreater(np.linalg.norm(res), 0.05)

    def test_output_shape(self):
        n_timesteps = 4
        n_cams = 2
        T, C = n_timesteps, n_cams
        n_res = T * C * 2 + max(0, T - 2) * 3
        xyz = np.tile(np.array([0.2, -0.1, 6.0], dtype=np.float64), T)
        params = np.concatenate([xyz, [0.0]])
        P_list = two_cameras()
        X_flat = params[: 3 * T]
        pixels = []
        for i in range(C):
            pixels.append(
                [
                    project_point(P_list[i], X_flat[3 * t : 3 * t + 3])
                    for t in range(T)
                ]
            )
        g = np.array([0.0, 0.0, -9.81])
        res = trajectory_residual(params, P_list, pixels, n_timesteps, dt=0.04, g=g)
        self.assertEqual(res.shape, (n_res,))


class TestOptimizeTrajectory(unittest.TestCase):
    def test_shape(self):
        n_timesteps = 5
        n_cams = 2
        dt = 0.05
        g = np.array([0.0, 0.0, -9.81])
        drag_true = 0.0
        P_list = two_cameras()
        a = g + drag_true
        t_idx = np.arange(n_timesteps, dtype=np.float64)
        z0 = np.array([0.0, 0.0, 8.0])
        X_vars = z0 + 0.5 * a * (t_idx * dt)[:, None] ** 2
        pixels = [
            [project_point(P_list[i], X_vars[t]) for t in range(n_timesteps)]
            for i in range(n_cams)
        ]

        X_opt, cov, drag_opt, result = optimize_trajectory(
            P_list,
            pixels,
            dt=dt,
            g=g,
            drag=0.0,
            pixel_sigma=1.0,
            physics_sigma=1.0,
            omega_phys=1.0,
        )
        self.assertEqual(X_opt.shape, (n_timesteps, 3))
        n_x = 3 * n_timesteps
        self.assertEqual(cov.shape, (n_x, n_x))
        self.assertIsInstance(drag_opt, float)
        self.assertTrue(result.success)
        np.testing.assert_allclose(X_opt, X_vars, rtol=0, atol=1e-5)
        np.testing.assert_allclose(drag_opt, drag_true, rtol=0, atol=1e-4)

    def test_recovers_trajectory(self):
        rng = np.random.default_rng(43)
        scene = main_style_scene(rng)
        P_list = scene["P_list"]
        pixels = scene["pixels"]
        traj_true = scene["traj_true"]
        g = scene["g"]
        dt = scene["dt"]
        drag_coef = scene["drag_coef"]
        n_timesteps = scene["n_timesteps"]
        n_cams = scene["n_cameras"]

        X_opt, cov, drag_opt, result = optimize_trajectory(
            P_list,
            pixels,
            dt=dt,
            g=g,
            drag=drag_coef,
            pixel_sigma=1.0,
            physics_sigma=1.0,
            omega_phys=1.0,
        )
        self.assertTrue(result.success)
        reproj_sq = []
        for t in range(n_timesteps):
            for i in range(n_cams):
                d = project_point(P_list[i], X_opt[t]) - pixels[i][t]
                reproj_sq.append(float(d @ d))
        mean_reproj = float(np.sqrt(np.mean(reproj_sq)))
        self.assertLess(mean_reproj, 3.5)

        mean_err = float(np.mean(np.linalg.norm(X_opt - traj_true, axis=1)))
        self.assertLess(mean_err, 2.5)
        np.testing.assert_allclose(drag_opt, drag_coef, rtol=0.35, atol=0.12)
        self.assertTrue(np.isfinite(cov).all())

    def test_recovers_noise(self):
        rng = np.random.default_rng(101)
        scene = main_style_scene(rng, pixel_noise_sigma=0.75)
        P_list, pixels, traj_true = scene["P_list"], scene["pixels"], scene["traj_true"]
        g, dt, drag_coef = scene["g"], scene["dt"], scene["drag_coef"]
        n_timesteps, n_cams = scene["n_timesteps"], scene["n_cameras"]

        X_opt, cov, drag_opt, result = optimize_trajectory(
            P_list,
            pixels,
            dt=dt,
            g=g,
            drag=drag_coef,
            pixel_sigma=1.0,
            physics_sigma=1.0,
            omega_phys=1.0,
        )
        self.assertTrue(result.success)
        mean_reproj = float(
            np.sqrt(
                np.mean(
                    [
                        np.sum((project_point(P_list[i], X_opt[t]) - pixels[i][t]) ** 2)
                        for t in range(n_timesteps)
                        for i in range(n_cams)
                    ]
                )
            )
        )
        self.assertLess(mean_reproj, 1.2)
        mean_err = float(np.mean(np.linalg.norm(X_opt - traj_true, axis=1)))
        self.assertLess(mean_err, 1.2)
        np.testing.assert_allclose(drag_opt, drag_coef, rtol=0.35, atol=0.15)
        self.assertTrue(np.isfinite(cov).all())

    def test_recovers_two_cameras(self):
        rng = np.random.default_rng(202)
        scene = main_style_scene(rng, n_cameras=2, pixel_noise_sigma=1.5)
        P_list, pixels, traj_true = scene["P_list"], scene["pixels"], scene["traj_true"]
        g, dt, drag_coef = scene["g"], scene["dt"], scene["drag_coef"]
        n_timesteps, n_cams = scene["n_timesteps"], scene["n_cameras"]

        X_opt, cov, drag_opt, result = optimize_trajectory(
            P_list,
            pixels,
            dt=dt,
            g=g,
            drag=drag_coef,
            pixel_sigma=1.0,
            physics_sigma=1.0,
            omega_phys=1.0,
        )
        self.assertTrue(result.success)
        mean_err = float(np.mean(np.linalg.norm(X_opt - traj_true, axis=1)))
        self.assertLess(mean_err, 2.0)
        np.testing.assert_allclose(drag_opt, drag_coef, rtol=0.35, atol=0.15)
        self.assertTrue(np.isfinite(cov).all())

    def test_recovers_high_drag(self):
        rng = np.random.default_rng(303)
        scene = main_style_scene(rng, drag_coef=0.45, pixel_noise_sigma=1.25)
        P_list, pixels, traj_true = scene["P_list"], scene["pixels"], scene["traj_true"]
        g, dt, drag_coef = scene["g"], scene["dt"], scene["drag_coef"]
        n_timesteps = scene["n_timesteps"]
        n_cams = scene["n_cameras"]

        X_opt, cov, drag_opt, result = optimize_trajectory(
            P_list,
            pixels,
            dt=dt,
            g=g,
            drag=drag_coef,
            pixel_sigma=1.0,
            physics_sigma=1.0,
            omega_phys=1.0,
        )
        self.assertTrue(result.success)
        mean_reproj = float(
            np.sqrt(
                np.mean(
                    [
                        np.sum((project_point(P_list[i], X_opt[t]) - pixels[i][t]) ** 2)
                        for t in range(n_timesteps)
                        for i in range(n_cams)
                    ]
                )
            )
        )
        self.assertLess(mean_reproj, 2.5)
        mean_err = float(np.mean(np.linalg.norm(X_opt - traj_true, axis=1)))
        self.assertLess(mean_err, 1.8)
        np.testing.assert_allclose(drag_opt, drag_coef, rtol=0.35, atol=0.15)
        self.assertTrue(np.isfinite(drag_opt))
        self.assertTrue(np.isfinite(cov).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
