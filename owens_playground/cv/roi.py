"""
Shot Put Tracker

Usage:
  python shotput_tracker.py --video path/to/video.mp4

Controls:
  SPACE - pause/resume
  Q     - quit
  R     - reset background model (useful if scene changes)

TODOs

Right now this is super downsampled which is totally ok

To give good (4k) esimates to Yash, we need to...
1. Note the landing location in px coords
2. Re run the pipeline for the set of frames the throw trajectory was visible in, but over the exact landing region at full resolution
3. Capture the exact frames and send to Yash

This isn't impossible and the fact that we can narrow down full res to a tiny ROI
  will make it actually perform better than on the analysis of the downsampled frame

Challenges
- The more frames Yash needs, potentially the larger ROI we need to capture at full res (slow)
- We need to keep trajectory history/frames in-memory until the throw is over

Stuff that will help
- We can make person on laptop specify which throwing event it is
  - This will let us fine-tune our object detection better
- We can make the person on the laptop specify a 'processing start' moment and a 'processing end' moment
- This way we can shoot high frame rate
  - Better confidence in object position across frames
  - Can give Yash good trajectory data over smaller ROI bc more frames
    - Smaller ROI means faster processing when at full res
  and be async.
  So we would start, and push all frames through a staged processing pipeline that is asynchronous
  When 'processing end' moment is hit, we can start the detailed full-res processing and be done in our 30s window
"""


import cv2
import numpy as np
import argparse
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Optional
import os

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
# DISPLAY_WIDTH  = 1920
# DISPLAY_HEIGHT = 1080
# PROCESS_WIDTH  = 960
# PROCESS_HEIGHT = 540

# Background subtractor
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 80
MOG2_DETECT_SHADOWS = False

MORPH_OPEN_KERNEL = 21
MORPH_CLOSE_KERNEL = 20
ROI_SIZE = 200
ROI_PADDING = 20

MIN_AREA = 8
MAX_AREA = 200
MAX_PERIMETER = 70
MIN_CIRCULARITY = 0.45
MAX_ASPECT_RATIO = 1.7

MAX_MISSED_FRAMES = 8
TRAIL_LENGTH = 600

# Background update interval for performance
BG_UPDATE_INTERVAL = 3

# from params import *
from tracking_funcs import *

def make_kalman():
    """Create Kalman filter for tracking"""
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-7
    kf.processNoiseCov[2, 2] = .10
    kf.processNoiseCov[3, 3] = .10
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 15.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    return kf


GRAVITY_PX_PER_FRAME2 = 0.6


@dataclass
class Tracker:
    kf: cv2.KalmanFilter = field(default_factory=make_kalman)
    initialized: bool = False
    missed: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    predicted: Optional[tuple] = None
    last_detection: Optional[tuple] = None

    def predict(self):
        """Run Kalman prediction step and inject gravity."""
        pred = self.kf.predict()
        self.kf.statePost[3] += GRAVITY_PX_PER_FRAME2
        x, y = int(pred[0][0]), int(pred[1][0])
        self.predicted = (x, y)
        return self.predicted

    def correct(self, cx, cy):
        """Feed a detection into the Kalman filter."""
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(meas)
        if not self.initialized:
            self.kf.statePost[0] = cx
            self.kf.statePost[1] = cy
            self.initialized = True
        self.trail.append((cx, cy))
        self.last_detection = (cx, cy)
        self.missed = 0

    def reset(self):
        self.__init__()


class ROIProcessor:
    """Handles ROI-based processing for speed optimization"""
    
    def __init__(self, process_width, process_height, roi_size=ROI_SIZE):
        self.process_width = process_width
        self.process_height = process_height
        self.roi_size = roi_size
        self.enabled = False
        self.roi_x = 0
        self.roi_y = 0
        self.roi_w = process_width
        self.roi_h = process_height
        self.stable_frame_count = 0
        
    def update_roi(self, predicted_pos, trail_length):
        """Update ROI based on predicted position"""
        if not predicted_pos:
            return False
            
        px, py = predicted_pos
        
        self.roi_x = max(0, px - self.roi_size // 2)
        self.roi_y = max(0, py - self.roi_size // 2)
        self.roi_w = min(self.roi_size, self.process_width - self.roi_x)
        self.roi_h = min(self.roi_size, self.process_height - self.roi_y)
        
        if trail_length > 10:
            self.stable_frame_count += 1
            if self.stable_frame_count > 5:
                self.enabled = True
        else:
            self.stable_frame_count = 0
            self.enabled = False
            
        return self.enabled
    
    def apply_roi(self, frame):
        """Extract ROI from frame"""
        if self.enabled and self.roi_w > 0 and self.roi_h > 0:
            return frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        return frame
    
    def map_to_original(self, x, y):
        """Map coordinates from ROI back to original frame"""
        if self.enabled:
            return x + self.roi_x, y + self.roi_y
        return x, y


def detect_candidates_optimized(mask):
    """
    Find circular blobs in a binary mask that could be the shot put.
    Returns list of (cx, cy, radius, circularity) tuples.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], [], [], [], [], []
    
    candidates = []
    rejected_area = []
    rejected_perimeter = []
    rejected_circularity = []
    rejected_aspect = []
    accepted = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            rejected_area.append(cnt)
            continue

        _, _, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > MAX_ASPECT_RATIO:
            rejected_aspect.append(cnt)
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or perimeter > MAX_PERIMETER:
            rejected_perimeter.append(cnt)
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            rejected_circularity.append(cnt)
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        candidates.append((int(cx), int(cy), int(radius), circularity))
        accepted.append(cnt)

    return candidates, accepted, rejected_area, rejected_perimeter, rejected_circularity, rejected_aspect


def pick_best_candidate(candidates, tracker: Tracker):
    """Pick the best candidate based on tracker prediction or circularity"""
    if not candidates:
        return None

    if tracker.initialized and tracker.predicted:
        px, py = tracker.predicted
        gate = 80
        gated = [c for c in candidates if np.hypot(c[0] - px, c[1] - py) < gate]
        if gated:
            return min(gated, key=lambda c: np.hypot(c[0] - px, c[1] - py))

    return max(candidates, key=lambda c: c[3]) if candidates else None


def make_candidate_vis(frame_gray, candidates, tracker):
    """Visualization of candidate blobs."""
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    for cx, cy, r, circ in candidates:
        cv2.circle(vis, (cx, cy), r, (0, 200, 255), 2)
        cv2.putText(vis, f"{circ:.2f}", (cx - 15, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    if tracker.predicted:
        px, py = tracker.predicted
        cv2.drawMarker(vis, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, (px, py), 80, (255, 0, 255), 1)
    return vis


def assemble_panels(panels: list[tuple[np.ndarray, str]]):
    """Tile frames into one display."""
    if not panels:
        return None
        
    num_rows = (len(panels) + 1) // 2
    target_h = DISPLAY_HEIGHT // num_rows
    target_w = DISPLAY_WIDTH if len(panels) == 1 else DISPLAY_WIDTH // 2

    def resize(f, target_w, target_h):
        if f is None or f.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
        h, w = f.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized = cv2.resize(f, (new_w, new_h))
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=0)

    def to_bgr(f):
        if len(f.shape) == 2:
            return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        return f

    resized_with_labels = []
    for f, label in panels:
        try:
            resized = resize(to_bgr(f), target_w, target_h)
            resized_with_labels.append((resized, label))
        except Exception:
            resized_with_labels.append((np.zeros((target_h, target_w, 3), dtype=np.uint8), label))

    for img, label in resized_with_labels:
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    rows = []
    for i in range(0, len(resized_with_labels), 2):
        l = resized_with_labels[i][0]
        r = resized_with_labels[i + 1][0] if i + 1 < len(resized_with_labels) else np.zeros_like(l)
        rows.append(np.hstack([l, r]))

    return np.vstack(rows)


def main(video_path: str, render_visualization: bool = True):
    # Check video file
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Processing resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    print(f"Visualization: {'ON' if render_visualization else 'OFF'}")

    # Initialize background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))

    tracker = Tracker()
    roi_processor = ROIProcessor(PROCESS_WIDTH, PROCESS_HEIGHT)
    
    paused = False
    frame_n = 0
    bg_update_counter = 0

    if render_visualization:
        cv2.namedWindow("Shot Put Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Shot Put Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    total_compute_start = time.perf_counter()
    total_frames_processed = 0

    # Initialize background model
    print("Initializing background model...")
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT)), cv2.COLOR_BGR2GRAY)
        bg_sub.apply(gray)
        frame_n += 1
    print("Background model initialized")

    while True:
        if not paused:
            frame_start = time.perf_counter_ns()
            
            ret, frame_orig = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_n += 1

            # Process frame
            try:
                frame_proc = cv2.resize(frame_orig, (PROCESS_WIDTH, PROCESS_HEIGHT))
                gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error processing frame {frame_n}: {e}")
                continue

            # Update ROI based on tracker
            if tracker.initialized and tracker.predicted:
                roi_processor.update_roi(tracker.predicted, len(tracker.trail))
            
            # Process with or without ROI
            if roi_processor.enabled:
                gray_roi = roi_processor.apply_roi(gray)
                
                if bg_update_counter % BG_UPDATE_INTERVAL == 0:
                    fg_mask = bg_sub.apply(gray_roi, learningRate=0.005)
                else:
                    fg_mask = bg_sub.apply(gray_roi, learningRate=0)
                bg_update_counter += 1
                
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                
                if fg_mask is not None and fg_mask.size > 0:
                    mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
                    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
                else:
                    mask_clean = np.zeros_like(gray_roi)

                candidates, accepted, r_area, r_per, r_circ, r_aspect = detect_candidates_optimized(mask_clean)
                
                # Map candidates back
                mapped_candidates = []
                for cx, cy, r, circ in candidates:
                    orig_x, orig_y = roi_processor.map_to_original(cx, cy)
                    mapped_candidates.append((orig_x, orig_y, r, circ))
            else:
                if bg_update_counter % BG_UPDATE_INTERVAL == 0:
                    fg_mask = bg_sub.apply(gray, learningRate=0.005)
                else:
                    fg_mask = bg_sub.apply(gray, learningRate=0)
                bg_update_counter += 1

                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

                if fg_mask is not None and fg_mask.size > 0:
                    mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
                    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
                else:
                    mask_clean = np.zeros_like(gray)

                candidates, accepted, r_area, r_per, r_circ, r_aspect = detect_candidates_optimized(mask_clean)
                mapped_candidates = candidates

            # Predict and correct
            if tracker.initialized:
                tracker.predict()

            best = pick_best_candidate(mapped_candidates, tracker)
            if best:
                cx, cy, radius, _ = best
                tracker.correct(cx, cy)
            else:
                tracker.missed += 1
                if tracker.missed > MAX_MISSED_FRAMES:
                    tracker.reset()
                    roi_processor.enabled = False
                    roi_processor.stable_frame_count = 0

            frame_stop = time.perf_counter_ns()
            elapsed_ms = (frame_stop - frame_start) / 1e6
            frame_rate = 1000 / elapsed_ms if elapsed_ms > 0 else float('inf')

            # Visualization
            if render_visualization:
                # Create output frame with predictions
                output_frame = frame_proc.copy()
                output_frame = draw_predictions(output_frame, tracker, best)
                
                # Draw ROI if enabled
                '''
                if roi_processor.enabled:
                    cv2.rectangle(output_frame,
                                 (roi_processor.roi_x, roi_processor.roi_y),
                                 (roi_processor.roi_x + roi_processor.roi_w,
                                  roi_processor.roi_y + roi_processor.roi_h),
                                 (0, 255, 255), 2)
                '''
                # Draw ROI (always show, but with different colors based on state)
                roi_color = (0, 255, 0) if roi_processor.enabled else (0, 0, 255)  # Green if active, red if inactive
                roi_label = "ROI ACTIVE" if roi_processor.enabled else "ROI INACTIVE"
                cv2.rectangle(output_frame,
                            (roi_processor.roi_x, roi_processor.roi_y),
                            (roi_processor.roi_x + roi_processor.roi_w,
                            roi_processor.roi_y + roi_processor.roi_h),
                            roi_color, 2)
                cv2.putText(output_frame, roi_label, 
                        (roi_processor.roi_x + 5, roi_processor.roi_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

                # Create candidate visualization
                cand_vis = make_candidate_vis(gray, candidates, tracker)
                
                # Create contour visualization
                contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if r_area:
                    cv2.drawContours(contour_vis, r_area, -1, (255, 0, 255), 1)
                if r_per:
                    cv2.drawContours(contour_vis, r_per, -1, (0, 0, 255), 1)
                if r_circ:
                    cv2.drawContours(contour_vis, r_circ, -1, (0, 165, 255), 1)
                if r_aspect:
                    cv2.drawContours(contour_vis, r_aspect, -1, (0, 255, 255), 1)
                if accepted:
                    cv2.drawContours(contour_vis, accepted, -1, (0, 255, 0), 2)
                
                # Assemble panels
                try:
                    quad = assemble_panels([
                        (frame_proc, "Original"),
                        (mask_clean, "Mask"),
                        (contour_vis, "Contours"),
                        (cand_vis, "Candidates"),
                        (output_frame, "Tracking"),
                    ])
                    
                    # Add FPS
                    cv2.putText(quad, f"FPS: {frame_rate:.1f}", 
                                (DISPLAY_WIDTH - 120, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow("Shot Put Tracker", quad)
                except Exception as e:
                    print(f"Visualization error: {e}")
                    cv2.imshow("Shot Put Tracker", cv2.resize(frame_proc, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
                time.sleep(.1) ## TODO: remove
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r'):
                    bg_sub = cv2.createBackgroundSubtractorMOG2(
                        history=MOG2_HISTORY,
                        varThreshold=MOG2_VAR_THRESHOLD,
                        detectShadows=MOG2_DETECT_SHADOWS,
                    )
                    tracker.reset()
                    roi_processor.enabled = False
                    roi_processor.stable_frame_count = 0
                    print("Background model and tracker reset.")
            
            total_frames_processed += 1

        else:
            # Paused state
            if render_visualization:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("Resumed")
                elif key == ord('r'):
                    bg_sub = cv2.createBackgroundSubtractorMOG2(
                        history=MOG2_HISTORY,
                        varThreshold=MOG2_VAR_THRESHOLD,
                        detectShadows=MOG2_DETECT_SHADOWS,
                    )
                    tracker.reset()
                    roi_processor.enabled = False
                    roi_processor.stable_frame_count = 0
                    print("Background model and tracker reset.")
            else:
                time.sleep(0.01)

    # Print summary
    total_compute_end = time.perf_counter()
    total_compute_time = total_compute_end - total_compute_start

    if total_frames_processed > 0:
        avg_time_per_frame = (total_compute_time / total_frames_processed) * 1000
        effective_fps = total_frames_processed / total_compute_time
        print(f"\n===== COMPUTE SUMMARY =====")
        print(f"Total frames processed: {total_frames_processed}")
        print(f"Average time per frame: {avg_time_per_frame:.2f} ms")
        print(f"Effective processing FPS: {effective_fps:.2f}")

    cap.release()
    if render_visualization:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot put tracker")
    parser.add_argument("--video", required=True, help="Path to .mp4 video file")
    parser.add_argument("--no-display", action="store_true", help="Disable visualization")
    parser.add_argument("--render", action="store_true", help="Enable visualization (default)")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")

    args = parser.parse_args()
    render = not (args.no_render or args.no_display)
    main(args.video, render_visualization=render)