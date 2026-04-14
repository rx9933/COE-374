import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
from tuned_cv import extract_trajectory_from_video


def _natural_cubic_spline_second_derivatives(t, y):
    n = len(t)
    if n < 3:
        return np.zeros(n, dtype=float)

    h = np.diff(t)
    alpha = np.zeros(n, dtype=float)
    alpha[1:-1] = (3.0 / h[1:]) * (y[2:] - y[1:-1]) - (3.0 / h[:-1]) * (y[1:-1] - y[:-2])

    l = np.ones(n, dtype=float)
    mu = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)

    for i in range(1, n - 1):
        l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n, dtype=float)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]

    return c


def _evaluate_cubic_spline(t, y, c, t_eval):
    y_eval = np.empty_like(t_eval, dtype=float)
    n = len(t)

    for idx, x_val in enumerate(t_eval):
        if x_val <= t[0]:
            i = 0
        elif x_val >= t[-1]:
            i = n - 2
        else:
            i = np.searchsorted(t, x_val) - 1

        h = t[i + 1] - t[i]
        if h == 0:
            y_eval[idx] = y[i]
            continue

        a = (t[i + 1] - x_val) / h
        b = (x_val - t[i]) / h
        y_eval[idx] = (
            a * y[i]
            + b * y[i + 1]
            + ((a**3 - a) * c[i] + (b**3 - b) * c[i + 1]) * (h**2) / 6.0
        )

    return y_eval


def _fit_cubic_spline(points, num_samples=200):
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return points

    t = np.linspace(0.0, 1.0, points.shape[0])
    x = points[:, 0]
    y = points[:, 1]
    c_x = _natural_cubic_spline_second_derivatives(t, x)
    c_y = _natural_cubic_spline_second_derivatives(t, y)

    t_sample = np.linspace(0.0, 1.0, num_samples)
    x_sample = _evaluate_cubic_spline(t, x, c_x, t_sample)
    y_sample = _evaluate_cubic_spline(t, y, c_y, t_sample)

    return np.vstack([x_sample, y_sample]).T


def interpolate_at_same_time_intervals(trajectory1, trajectory2, num_points=100):
    """
    Interpolate both trajectories at the same normalized time intervals.
    
    Args:
        trajectory1: numpy array of shape (N1, 2) - first trajectory points
        trajectory2: numpy array of shape (N2, 2) - second trajectory points
        num_points: number of interpolated points to generate for each trajectory
    
    Returns:
        interp1: interpolated points for trajectory 1 at same time intervals
        interp2: interpolated points for trajectory 2 at same time intervals
    """
    # Create normalized time parameter (0 to 1) for each trajectory
    t1 = np.linspace(0.0, 1.0, len(trajectory1))
    t2 = np.linspace(0.0, 1.0, len(trajectory2))
    
    # Create common time points
    t_common = np.linspace(0.0, 1.0, num_points)
    
    # Fit cubic splines for trajectory 1
    cs_x1 = CubicSpline(t1, trajectory1[:, 0], bc_type='natural')
    cs_y1 = CubicSpline(t1, trajectory1[:, 1], bc_type='natural')
    
    # Fit cubic splines for trajectory 2
    cs_x2 = CubicSpline(t2, trajectory2[:, 0], bc_type='natural')
    cs_y2 = CubicSpline(t2, trajectory2[:, 1], bc_type='natural')
    
    # Evaluate at common time points
    interp1 = np.column_stack([cs_x1(t_common), cs_y1(t_common)])
    interp2 = np.column_stack([cs_x2(t_common), cs_y2(t_common)])
    
    return interp1, interp2, t_common


def plot_two_throws_with_common_interpolation(video_path1: str, video_path2: str, 
                                              output_path: str = "throw_comparison_with_interpolation.png",
                                              num_interp_points=100):
    """
    Plot two throw trajectories with cubic spline interpolation at the same time intervals.
    Shows both the original tracked points and the interpolated points at common times.
    """
    # Extract trajectories
    positions1, detected1, fps1, _, trail1 = extract_trajectory_from_video(video_path1)
    positions2, detected2, fps2, _, trail2 = extract_trajectory_from_video(video_path2)

    tracked1 = np.asarray(trail1, dtype=float)
    tracked2 = np.asarray(trail2, dtype=float)

    if tracked1.size == 0 or tracked2.size == 0:
        print("One of the throws has no trajectory data.")
        return

    # Filter detection status
    def filter_detection_status(positions, detected, tracked):
        if len(positions) == len(detected):
            filtered = np.asarray([d for p, d in zip(positions, detected) if p is not None], dtype=bool)
            if len(filtered) != len(tracked):
                return np.ones(len(tracked), dtype=bool)
            return filtered
        return np.ones(len(tracked), dtype=bool)

    detected1 = filter_detection_status(positions1, detected1, tracked1)
    detected2 = filter_detection_status(positions2, detected2, tracked2)

    # Interpolate both trajectories at the same normalized time intervals
    interp1, interp2, t_common = interpolate_at_same_time_intervals(tracked1, tracked2, num_interp_points)
    
    # Calculate distance between corresponding interpolated points
    distances = np.sqrt(np.sum((interp1 - interp2)**2, axis=1))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Main trajectory plot
    ax1 = plt.subplot(2, 2, (1, 2))
    
    # Plot the full spline trajectories
    spline1_full = _fit_cubic_spline(tracked1, num_samples=200)
    spline2_full = _fit_cubic_spline(tracked2, num_samples=200)
    
    ax1.plot(spline1_full[:, 0], spline1_full[:, 1], 'r-', linewidth=2.5, alpha=0.7, label='Throw 1 trajectory')
    ax1.plot(spline2_full[:, 0], spline2_full[:, 1], 'b-', linewidth=2.5, alpha=0.7, label='Throw 2 trajectory')
    
    # Plot original tracked points
    ax1.scatter(tracked1[detected1, 0], tracked1[detected1, 1], 
               c='red', s=30, alpha=0.5, label='Throw 1 detected')
    ax1.scatter(tracked2[detected2, 0], tracked2[detected2, 1], 
               c='blue', s=30, alpha=0.5, label='Throw 2 detected')
    
    # Plot interpolated points at common times
    # Color by normalized time (t_common)
    scatter1 = ax1.scatter(interp1[:, 0], interp1[:, 1], 
                          c=t_common, cmap='RdYlGn', s=60, 
                          marker='o', edgecolors='black', linewidth=1.5,
                          label='Throw 1 interpolated', vmin=0, vmax=1)
    scatter2 = ax1.scatter(interp2[:, 0], interp2[:, 1], 
                          c=t_common, cmap='RdYlGn', s=60, 
                          marker='s', edgecolors='black', linewidth=1.5,
                          label='Throw 2 interpolated', vmin=0, vmax=1)
    
    # Connect corresponding points with lines
    for i in range(0, len(interp1), max(1, num_interp_points // 20)):  # Show every Nth line to avoid clutter
        ax1.plot([interp1[i, 0], interp2[i, 0]], 
                [interp1[i, 1], interp2[i, 1]], 
                'gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Mark start and end points
    ax1.scatter(tracked1[0, 0], tracked1[0, 1], c='darkred', s=200, 
               marker='*', label='Throw 1 start', zorder=5, edgecolors='black', linewidth=2)
    ax1.scatter(tracked1[-1, 0], tracked1[-1, 1], c='red', s=200, 
               marker='*', label='Throw 1 end', zorder=5, edgecolors='black', linewidth=2)
    ax1.scatter(tracked2[0, 0], tracked2[0, 1], c='darkblue', s=200, 
               marker='*', label='Throw 2 start', zorder=5, edgecolors='black', linewidth=2)
    ax1.scatter(tracked2[-1, 0], tracked2[-1, 1], c='blue', s=200, 
               marker='*', label='Throw 2 end', zorder=5, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.set_title('Throw Comparison with Common-Time Interpolation', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # Add colorbar for normalized time
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label('Normalized Time (0=start, 1=end)', fontsize=10)
   
    
    # Add statistics
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    final_distance = distances[-1]

    
    # Trajectory information table
    ax2 = plt.subplot(2, 2, 4)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Prepare statistics
    stats_data = [
        ['Metric', 'Throw 1', 'Throw 2'],
        ['Total frames', f'{len(tracked1)}', f'{len(tracked2)}'],
        ['Duration (s)', f'{len(tracked1)/fps1:.2f}', f'{len(tracked2)/fps2:.2f}'],
        ['X range', f'{tracked1[:,0].min():.1f}-{tracked1[:,0].max():.1f}', 
         f'{tracked2[:,0].min():.1f}-{tracked2[:,0].max():.1f}'],
        ['Y range', f'{tracked1[:,1].min():.1f}-{tracked1[:,1].max():.1f}', 
         f'{tracked2[:,1].min():.1f}-{tracked2[:,1].max():.1f}'],
        ['Detected %', f'{np.sum(detected1)/len(detected1)*100:.1f}%', 
         f'{np.sum(detected2)/len(detected2)*100:.1f}%'],
    ]
    
    # Create table
    table = ax2.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(len(stats_data[0])):
            if i == 0:
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    table[(i, j)].set_facecolor('#e6e6e6')
                else:
                    table[(i, j)].set_facecolor('#f5f5f5')
    
    ax2.set_title('Trajectory Statistics', fontsize=12, pad=20)
    
    plt.suptitle(f'Throw Comparison with {num_interp_points} Common-Time Interpolation Points\n'
                f'Line connecting corresponding time points | Color indicates normalized time',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Interpolation Results ===")
    print(f"Number of common interpolation points: {num_interp_points}")
    print(f"Mean distance between trajectories: {mean_distance:.2f} pixels")
    print(f"Max distance between trajectories: {max_distance:.2f} pixels")
    print(f"Min distance between trajectories: {min_distance:.2f} pixels")
    print(f"Final separation distance: {final_distance:.2f} pixels")
    print(f"Comparison plot saved to: {output_path}")
    
    return interp1, interp2, t_common, distances


# Example usage
if __name__ == "__main__":
    throw0 = "Video_Camera_Processing/throws/Arushi_throw_0.mp4"
    throw1 = "Video_Camera_Processing/throws/Arushi_throw_1.mp4"
    
    # Plot with common interpolation points
    interp1, interp2, t_common, distances = plot_two_throws_with_common_interpolation(
        throw0, throw1, 
        output_path="throw_comparison_interpolated.png",
        num_interp_points=100
    )