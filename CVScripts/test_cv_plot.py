import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the tracking function from your existing code
from tuned_cv import extract_trajectory_from_video

def plot_trajectory_only(video_path: str, output_path: str = "trajectory_plot.png"):
    """
    Plot ONLY the tracked trajectory 
    Uses extract_trajectory_from_video 
    Does NOT plot ROI centers or any other markers.
    """
    # Extract the trajectory currently represented by tuned_cv's red trail
    positions, detected, fps, (width, height), trail = extract_trajectory_from_video(video_path)
    
    # The returned trail is already the red-line trajectory from tuned_cv
    tracked_positions = trail
    detection_status = detected
    
    if len(tracked_positions) == 0:
        print("No trajectory data found!")
        return
    
    tracked_array = np.array(tracked_positions)
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Complete trajectory (the red line)
    ax1.plot(tracked_array[:, 0], tracked_array[:, 1], 'r-', linewidth=2.5, label='Tracked Trajectory')
    
    # Mark start and end points
    ax1.scatter(tracked_array[0, 0], tracked_array[0, 1], 
               c='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=2)
    ax1.scatter(tracked_array[-1, 0], tracked_array[-1, 1], 
               c='red', s=200, marker='s', label='End', zorder=5, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.set_title('Shot Put Trajectory (Red Line from Visualization)', fontsize=14)
    ax1.invert_yaxis()  # Match image coordinates
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics box
    stats_text = f'Total tracked points: {len(tracked_positions)}\n'
    stats_text += f'Detected: {sum(detection_status)}\n'
    stats_text += f'Predicted: {len(detection_status) - sum(detection_status)}\n'
    stats_text += f'FPS: {fps:.1f}\n'
    stats_text += f'Duration: {len(tracked_positions)/fps:.2f}s\n'
    stats_text += f'X range: {tracked_array[:, 0].min():.1f} - {tracked_array[:, 0].max():.1f}\n'
    stats_text += f'Y range: {tracked_array[:, 1].min():.1f} - {tracked_array[:, 1].max():.1f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: X and Y coordinates over time
    frames = range(len(tracked_positions))
    ax2.plot(frames, tracked_array[:, 0], 'r-', label='X coordinate', alpha=0.7, linewidth=2)
    ax2.plot(frames, tracked_array[:, 1], 'b-', label='Y coordinate', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Pixel Coordinate', fontsize=12)
    ax2.set_title('Position vs Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection vs Prediction visualization
    colors = ['green' if det else 'orange' for det in detection_status]
    labels = ['Detected' if det else 'Predicted' for det in detection_status]
    
    for i in range(len(tracked_positions)):
        ax3.scatter(tracked_array[i, 0], tracked_array[i, 1], 
                   c=colors[i], s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax3.plot(tracked_array[:, 0], tracked_array[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax3.set_xlabel('X (pixels)', fontsize=12)
    ax3.set_ylabel('Y (pixels)', fontsize=12)
    ax3.set_title('Detection Type Color Coding\nGreen=Detected, Orange=Predicted', fontsize=14)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    
    # Create custom legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.6, label='Detected'),
                      Patch(facecolor='orange', alpha=0.6, label='Predicted')]
    ax3.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Trajectory Statistics ===")
    print(f"Total frames processed: {len(positions)}")
    print(f"Tracked positions (red line points): {len(tracked_positions)}")
    print(f"  - Detected (green): {sum(detection_status)}")
    print(f"  - Predicted (orange): {len(detection_status) - sum(detection_status)}")
    print(f"Trajectory plot saved to: {output_path}")
    
    return tracked_array, detection_status, fps



# Example usage
if __name__ == "__main__":
    # Replace with your video path
    video_path = "Video_Camera_Processing/throws/Arushi_throw_0.mp4"
    
    # Plot just the trajectory
    positions, detected, fps = plot_trajectory_only(video_path)
    
    # Plot trajectory overlaid on video frame
    # plot_with_overlay(video_path)