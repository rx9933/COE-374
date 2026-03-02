"""
Image to 3D position estimation pipeline.

1. Loads images per (trial, camera, frame) and runs CVScripts/single_frame_processing and 
MathScripts/Physics_Triangulation_No_Camera_Conditioning to estimate and plot 3D trajectory.

Usage:
  python photo_to_position.py --images-dir /path/to/images --trial 0 --n-cameras 3 --n-frames 25 \\
      --P-list P.npy --dt 0.033 --g 0 0 -9.81 --out trajectory_3d.png

  Expects files: {images_dir}/trial{trial}_camera{cam}_frame{frame}.png
  with cam in [0, n_cameras), frame in [0, n_frames).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "MathScripts"))

from CVScripts.single_frame_detection import detect_shotput
from MathScripts.Physics_Triangulation_No_Camera_Conditioning import optimize_trajectory


def collect_pixels_from_images(images_dir, trial, n_cameras, n_frames, conf=0.2, imgsz=1080):
    images_dir = Path(images_dir)
    pixels = []
    for cam in range(n_cameras):
        row = []
        for frame in range(n_frames):
            path = images_dir / f"trial{trial}_camera{cam}_frame{frame}.png"
            if not path.exists():
                raise FileNotFoundError(f"Missing image: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Could not read image: {path}")
            result = detect_shotput(img, conf=conf, imgsz=imgsz)
            if result is None:
                raise ValueError(f"No projectile detected: trial{trial} camera{cam} frame{frame}.")
            (x1, y1, x2, y2), _ = result
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            row.append(np.array([cx, cy], dtype=np.float64))
        pixels.append(row)
    return pixels


def run_pipeline(images_dir, trial, n_cameras, n_frames, P_list, dt, g = [0, 0, -9.81], pixel_sigma=1.0, physics_sigma=0.01, conf=0.2, imgsz=1080):
    """
    Run detection on each image and triangulate to 3D.

    Args:
        images_dir: Directory containing trialX_cameraI_frameY.png images.
        trial: Trial index.
        n_cameras: Number of cameras.
        n_frames: Number of frames.
        P_list: List of 3x4 projection matrices (one per camera) as numpy arrays.
        dt: Time step in seconds.
        g: Gravity vector (3,).
        pixel_sigma: Observation noise scale for pixels.
        physics_sigma: Observation noise scale for physics residual.
        conf: YOLO confidence threshold.
        imgsz: YOLO inference resolution.
    Returns:
        X_opt: (n_frames, 3) estimated 3D positions.
        cov: (3*n_frames, 3*n_frames) trajectory covariance.
    """

    if n_cameras < 2:
        raise ValueError("At least 2 cameras required.")
    if len(P_list) != n_cameras:
        raise ValueError(f"P_list has {len(P_list)} cameras but n_cameras={n_cameras}.")
    if n_frames < 3:
        raise ValueError("Need at least 3 frames for physics triangulation.")

    pixels = collect_pixels_from_images(images_dir, trial, n_cameras, n_frames, conf=conf, imgsz=imgsz)

    X_opt, cov, _ = optimize_trajectory(
        P_list, pixels, dt=dt, g=np.asarray(g, dtype=np.float64), drag=0.0,
        pixel_sigma=pixel_sigma, physics_sigma=physics_sigma,
    )
    return X_opt, cov


def plot_3d_trajectory(X_opt, cov=None, out_path="trajectory_3d.png"):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Install matplotlib to plot: pip install matplotlib")
        return
    n = len(X_opt)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], "b-o", markersize=4, label="Estimated 3D trajectory")
    if cov is not None and np.all(np.isfinite(cov)):
        max_std = np.max(np.sqrt(np.diag(cov)))
        scale = 0.3 / max_std if max_std > 1.0 else 1.0
        for t in range(0, n, max(1, n // 15)):
            cov_t = cov[3 * t : 3 * t + 3, 3 * t : 3 * t + 3]
            if np.any(np.isnan(cov_t)) or np.any(np.linalg.eigvalsh(cov_t) <= 0):
                continue
            eigs, Q = np.linalg.eigh(cov_t)
            eigs = np.maximum(eigs, 1e-12)
            radii = scale * np.sqrt(eigs)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) @ (Q * radii).T + X_opt[t]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red", alpha=0.15, s=1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Images (trialX_cameraI_frameY.png) to 3D trajectory"
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing trialX_cameraI_frameY.png")
    parser.add_argument("--trial", type=int, required=True, help="Trial index X in filename")
    parser.add_argument("--n-cameras", type=int, required=True, help="Number of cameras (I = 0 .. n_cameras-1)")
    parser.add_argument("--n-frames", type=int, required=True, help="Number of frames (Y = 0 .. n_frames-1)")
    parser.add_argument("--P-list", required=True, help="Path to .npy file: array of shape (n_cameras, 3, 4)")
    parser.add_argument("--dt", type=float, default=0.016666666666666666, help="Time step in seconds")
    parser.add_argument("--g", nargs=3, type=float, default=[0, 0, -9.81], help="Gravity vector, e.g. 0 0 -9.81")
    parser.add_argument("--pixel-sigma", type=float, default=1.0, help="Pixel observation noise scale")
    parser.add_argument("--physics-sigma", type=float, default=0.01, help="Physics residual noise scale")
    parser.add_argument("--conf", type=float, default=0.2, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1080, help="YOLO inference resolution")
    parser.add_argument("--out", type=str, default="trajectory_3d.png", help="Output plot path")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    P_list = np.load(args.P_list)
    if P_list.ndim != 3 or P_list.shape[1] != 3 or P_list.shape[2] != 4:
        raise ValueError("P_list must have shape (n_cameras, 3, 4).")
    P_list = [P_list[i] for i in range(len(P_list))]

    X_opt, cov = run_pipeline(
        images_dir=images_dir,
        trial=args.trial,
        n_cameras=args.n_cameras,
        n_frames=args.n_frames,
        P_list=P_list,
        dt=args.dt,
        g=args.g,
        pixel_sigma=args.pixel_sigma,
        physics_sigma=args.physics_sigma,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    print(f"Estimated 3D trajectory: {len(X_opt)} frames, dt={args.dt} s")
    print(f"\tx range: [{X_opt[:, 0].min():.3f}, {X_opt[:, 0].max():.3f}] m")
    print(f"\ty range: [{X_opt[:, 1].min():.3f}, {X_opt[:, 1].max():.3f}] m")
    print(f"\tz range: [{X_opt[:, 2].min():.3f}, {X_opt[:, 2].max():.3f}] m")
    plot_3d_trajectory(X_opt, cov=cov, out_path=args.out)


if __name__ == "__main__":
    main()
