"""
Shot Put Tracker

Usage:
  python shotput_tracker.py --video path/to/video.mp4

Controls:
  SPACE - pause/resume
  Q     - quit
  R     - reset background model (useful if scene changes)
  LEFT  - go back one frame (when paused)
  RIGHT - go forward one frame (when paused)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Optional, Tuple, List

DISPLAY_WIDTH  = 1920
DISPLAY_HEIGHT = 1080
PROCESS_WIDTH  = 960
PROCESS_HEIGHT = 540

# Background subtractor
MOG2_HISTORY        = 300   # frames to build background model
MOG2_VAR_THRESHOLD  = 60    # lower = more sensitive; raise if noisy background
MOG2_DETECT_SHADOWS = False

MORPH_OPEN_KERNEL  = 3   # removes small noise blobs
MORPH_CLOSE_KERNEL = 40  # fills holes inside the shot put blob

# ROI tracking after initialization
ROI_SIZE = 100  # pixels in process-space (width and height of ROI)
ROI_PADDING = 20  # extra padding around predicted position
MIN_ROI_SIZE = 100  # minimum ROI size when not initialized

# Consistency check parameters
CONSISTENCY_WINDOW = 3  # number of frames to check for consistency
MAX_DISTANCE_VARIATION = 15  # maximum allowed variation in distances (pixels)
MIN_CONSISTENT_DETECTIONS = 4  # REDUCED from 5 to 2 - faster ROI activation

MIN_AREA            = 8    # px^2 — ignore tiny noise
MAX_AREA            = 50   # px^2 — ignore huge regions
MAX_PERIMETER       = 70    # px — ignore very large contours (athlete body)
MIN_CIRCULARITY     = 0.63   # 1.0 = perfect circle; lower catches slight blur61
MAX_ASPECT_RATIO    = 1.7    # width/height of bounding rect; rejects lines

MAX_MISSED_FRAMES   = 8     # frames without detection before tracker resets
TRAIL_LENGTH        = 60    # how many past positions to draw as trail

def make_kalman():
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
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.processNoiseCov[2, 2] = 1.0
    kf.processNoiseCov[3, 3] = 1.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0 
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

GRAVITY_PX_PER_FRAME2 = 0.01

@dataclass
class Tracker:
    kf: cv2.KalmanFilter = field(default_factory=make_kalman)
    initialized: bool = False
    missed: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    predicted: Optional[tuple] = None
    last_position: Optional[tuple] = None
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=CONSISTENCY_WINDOW))
    use_roi: bool = False
    # Track number of consistent detections
    consistent_detection_count: int = 0

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
        
        self.last_position = (cx, cy)
        self.trail.append((cx, cy))
        self.recent_positions.append((cx, cy))
        self.missed = 0
        self.update_roi_flag()

    def update_roi_flag(self):
        """Check if recent positions are consistent enough to use ROI."""
        if len(self.recent_positions) < MIN_CONSISTENT_DETECTIONS:
            self.use_roi = False
            self.consistent_detection_count = 0
            return
        
        # Calculate distances between consecutive positions
        positions = list(self.recent_positions)
        distances = []
        for i in range(1, len(positions)):
            dist = np.hypot(positions[i][0] - positions[i-1][0], 
                           positions[i][1] - positions[i-1][1])
            distances.append(dist)
        
        if len(distances) < 2:
            self.use_roi = False
            return
        
        # Check if distances are consistent (not exploding)
        max_dist = max(distances)
        min_dist = min(distances)
        
        # Variation should be within threshold
        if (max_dist - min_dist) < MAX_DISTANCE_VARIATION and max_dist < MAX_DISTANCE_VARIATION * 2:
            self.consistent_detection_count += 1
            # Activate ROI after MIN_CONSISTENT_DETECTIONS consistent detections
            if self.consistent_detection_count >= MIN_CONSISTENT_DETECTIONS:
                self.use_roi = True
        else:
            self.consistent_detection_count = 0
            self.use_roi = False

    def reset(self):
        self.__init__()

def get_roi_from_prediction(tracker: Tracker, frame_shape: Tuple[int, int]) -> Tuple[slice, slice, Tuple[int, int]]:
    """
    Get ROI slices based on tracker prediction.
    Returns (y_slice, x_slice, offset) where offset is (top, left) for mapping back to full frame.
    """
    h, w = frame_shape[:2]
    
    if tracker.initialized and tracker.predicted:
        # Always use predicted position for ROI center if we have consistent detections
        # This helps even when use_roi flag is False
        cx, cy = tracker.predicted
        # Dynamically adjust ROI size based on velocity
        if len(tracker.trail) > 2:
            vx = tracker.kf.statePost[2][0]
            vy = tracker.kf.statePost[3][0]
            speed = np.hypot(vx, vy)
            dynamic_roi = int(ROI_SIZE * (1.0 + speed / 20.0))
            roi_size = min(ROI_SIZE * 2, max(ROI_SIZE, dynamic_roi))
        else:
            roi_size = ROI_SIZE
    else:
        # Fallback to center of frame
        cx, cy = w // 2, h // 2
        roi_size = MIN_ROI_SIZE
    
    # Calculate ROI boundaries
    left = max(0, cx - roi_size // 2 - ROI_PADDING)
    right = min(w, cx + roi_size // 2 + ROI_PADDING)
    top = max(0, cy - roi_size // 2 - ROI_PADDING)
    bottom = min(h, cy + roi_size // 2 + ROI_PADDING)
    
    return (slice(top, bottom), slice(left, right)), (top, left)

def pick_best_candidate_with_roi_priority(candidates, tracker: Tracker, roi_slice, roi_offset):
    """
    Pick the best candidate with priority given to candidates within ROI,
    especially after consistent detections.
    """
    if not candidates:
        return None
    
    # Extract ROI boundaries
    top, left = roi_offset
    bottom = top + (roi_slice[0].stop - roi_slice[0].start)
    right = left + (roi_slice[1].stop - roi_slice[1].start)
    
    # Separate candidates into ROI and non-ROI
    roi_candidates = []
    non_roi_candidates = []
    
    for c in candidates:
        cx, cy, _, _ = c
        if left <= cx <= right and top <= cy <= bottom:
            roi_candidates.append(c)
        else:
            non_roi_candidates.append(c)
    
    # If we have consistent detections (even if use_roi flag is False), prefer ROI candidates
    if tracker.consistent_detection_count >= MIN_CONSISTENT_DETECTIONS:
        # First try to find a good candidate within ROI
        if roi_candidates:
            # Use scoring within ROI candidates
            if tracker.initialized and tracker.predicted:
                return min(roi_candidates, key=lambda c: candidate_score(c, tracker))
            else:
                return max(roi_candidates, key=lambda c: c[3])
        
        # If no ROI candidates but we have non-ROI candidates, only accept if very close
        if non_roi_candidates and tracker.predicted:
            px, py = tracker.predicted
            # Only accept non-ROI candidates if they're very close to prediction
            close_candidates = [c for c in non_roi_candidates 
                               if np.hypot(c[0] - px, c[1] - py) < 20]  # Stricter gate
            if close_candidates:
                return min(close_candidates, key=lambda c: candidate_score(c, tracker))
            return None  # Reject far-away candidates
    
    # No consistent detections yet - use original logic
    if tracker.initialized and tracker.predicted:
        px, py = tracker.predicted
        gate = 80
        gated = [c for c in candidates if np.hypot(c[0] - px, c[1] - py) < gate]
        
        if gated and len(tracker.trail) > 2:
            avg_speed = np.mean([
                np.hypot(tracker.trail[i][0]-tracker.trail[i-1][0], 
                        tracker.trail[i][1]-tracker.trail[i-1][1])
                for i in range(1, len(tracker.trail))
            ])
            max_allowed = avg_speed * 2.5
            speed_gated = [c for c in gated if np.hypot(c[0] - px, c[1] - py) <= max_allowed]
            if speed_gated:
                gated = speed_gated
        if gated:
            return min(gated, key=lambda c: candidate_score(c, tracker))
    
    # If no tracker yet or no prediction, pick most circular
    return max(candidates, key=lambda c: c[3]) if candidates else None

def candidate_score(candidate, tracker: Tracker):
    """Lower score = better candidate."""
    cx, cy, r, circ = candidate

    if tracker.predicted is None:
        return -circ

    px, py = tracker.predicted
    dist = np.hypot(cx - px, cy - py)

    vx = tracker.kf.statePost[2][0]
    vy = tracker.kf.statePost[3][0]
    expected_x = px + vx
    expected_y = py + vy
    vel_err = np.hypot(cx - expected_x, cy - expected_y)

    if len(tracker.trail) > 0:
        size_err = abs(r - 6)
    else:
        size_err = 0

    score = dist + 0.5 * vel_err + 0.5 * size_err - 2 * circ
    return score

def detect_candidates_in_roi(mask, roi_slice, roi_offset):
    """Find circular blobs in a binary mask within the specified ROI."""
    top, left = roi_offset
    roi_mask = mask[roi_slice]
    
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])
            cx = cx_roi + left
            cy = cy_roi + top
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            candidates.append((int(cx), int(cy), int(radius), circularity))
            accepted.append(cnt)

    return candidates, accepted, rejected_area, rejected_perimeter, rejected_circularity, rejected_aspect

def detect_candidates_full(mask):
    """Find circular blobs in a binary mask across the full frame."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            candidates.append((int(cx), int(cy), int(radius), circularity))
            accepted.append(cnt)

    return candidates, accepted, rejected_area, rejected_perimeter, rejected_circularity, rejected_aspect

def draw_trail(frame, trail):
    pts = list(trail)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 3)

def draw_roi(frame, roi_slice, roi_offset, use_roi=True, consistent_count=0):
    """Draw ROI rectangle on frame with different color based on usage and consistency."""
    top, left = roi_offset
    bottom = top + (roi_slice[0].stop - roi_slice[0].start)
    right = left + (roi_slice[1].stop - roi_slice[1].start)
    
    # Use different color based on consistency and ROI status
    if consistent_count >= MIN_CONSISTENT_DETECTIONS:
        if use_roi:
            color = (0, 255, 0)  # Bright green - fully active
            label = f"ROI (ACTIVE) - {consistent_count} consistent"
        else:
            color = (0, 255, 255)  # Yellow - consistent but ROI off
            label = f"ROI (PRIORITY) - {consistent_count} consistent"
    else:
        color = (0, 0, 255)  # Red - not consistent
        label = f"ROI (INACTIVE) - {consistent_count}/{MIN_CONSISTENT_DETECTIONS}"
    
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, label, (left + 5, top + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def make_candidate_vis(frame_gray, candidates, tracker, roi_slice=None, roi_offset=None):
    """Visualization of candidate blobs."""
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    
    if roi_slice is not None and roi_offset is not None:
        top, left = roi_offset
        bottom = top + (roi_slice[0].stop - roi_slice[0].start)
        right = left + (roi_slice[1].stop - roi_slice[1].start)
        
        if tracker.consistent_detection_count >= MIN_CONSISTENT_DETECTIONS:
            color = (0, 255, 255) if not tracker.use_roi else (0, 255, 0)
            label = f"ROI (PRIORITY)" if not tracker.use_roi else "ROI (ACTIVE)"
        else:
            color = (0, 0, 255)
            label = "ROI (INACTIVE)"
        
        cv2.rectangle(vis, (left, top), (right, bottom), color, 2)
        cv2.putText(vis, label, (left + 5, top + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw candidates with different colors for ROI vs non-ROI
    for cx, cy, r, circ in candidates:
        # Check if candidate is in ROI
        in_roi = False
        if roi_slice is not None and roi_offset is not None:
            top, left = roi_offset
            bottom = top + (roi_slice[0].stop - roi_slice[0].start)
            right = left + (roi_slice[1].stop - roi_slice[1].start)
            in_roi = (left <= cx <= right and top <= cy <= bottom)
        
        color = (0, 255, 0) if in_roi else (0, 100, 255)  # Green for ROI, orange for outside
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.putText(vis, f"{circ:.2f}", (cx - 15, cy - r - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    if tracker.predicted:
        px, py = tracker.predicted
        cv2.drawMarker(vis, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, (px, py), 80, (255, 0, 255), 1)
    
    # Add consistency info
    status_y = 50
    cv2.putText(vis, f"Consistent: {tracker.consistent_detection_count}/{MIN_CONSISTENT_DETECTIONS}", 
                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0) if tracker.consistent_detection_count >= MIN_CONSISTENT_DETECTIONS else (0, 0, 255), 1)
    
    return vis

def assemble_panels(panels: List[Tuple[np.ndarray, str]]):
    """Tile frames into one display."""
    num_rows = (len(panels) + 1) // 2
    target_h = DISPLAY_HEIGHT // num_rows
    target_w = DISPLAY_WIDTH if len(panels) == 1 else DISPLAY_WIDTH // 2

    def resize(f, target_w, target_h):
        h, w = f.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(f, (new_w, new_h))
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    def to_bgr(f):
        if len(f.shape) == 2:
            return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        return f

    resized_with_labels = [(resize(to_bgr(f), target_w, target_h), label) for f, label in panels]

    for img, label in resized_with_labels:
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    rows = []
    for i in range(0, len(resized_with_labels), 2):
        l = resized_with_labels[i][0]
        r = resized_with_labels[i+1][0] if i+1 < len(resized_with_labels) else np.zeros_like(l)
        row = np.hstack([l, r])
        rows.append(row)

    return np.vstack(rows)

def main(video_path: str, render_visualization: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {fps:.1f}fps")
    print(f"Visualization: {'ON' if render_visualization else 'OFF'}")
    print(f"ROI will prioritize candidates after {MIN_CONSISTENT_DETECTIONS} consistent detections")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))

    tracker = Tracker()
    paused = False
    frame_n = 0

    if render_visualization:
        cv2.namedWindow("Shot Put Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Shot Put Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('roi_priority.mp4', fourcc, 30.0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    total_compute_start = time.perf_counter()
    total_frames_processed = 0
    
    while True:
        if not paused:
            frame_start = time.perf_counter_ns()
            ret, frame_orig = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_n += 1

            frame_proc = cv2.resize(frame_orig, (PROCESS_WIDTH, PROCESS_HEIGHT))
            gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

            # Get ROI based on tracker prediction
            roi_slice, roi_offset = get_roi_from_prediction(tracker, gray.shape)
            
            # Apply background subtraction - always do full frame but track ROI separately
            fg_mask = bg_sub.apply(gray)
            
            # Threshold to binary
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
            
            # Detect candidates in full frame
            candidates, accepted_candidates, r_area, r_per, r_circ, r_aspect = detect_candidates_full(mask_clean)
            
            # Kalman prediction
            if tracker.initialized:
                tracker.predict()

            # Pick best candidate with ROI priority
            best = pick_best_candidate_with_roi_priority(candidates, tracker, roi_slice, roi_offset)
            
            if best:
                cx, cy, radius, _ = best
                tracker.correct(cx, cy)
                tracker.missed = 0
            else:
                tracker.missed += 1
                if tracker.missed > MAX_MISSED_FRAMES:
                    tracker.reset()
            
            # Visualization
            if render_visualization:
                contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_vis, r_area, -1, (255, 0, 255), 2)
                cv2.drawContours(contour_vis, r_per, -1, (0, 0, 255), 2)
                cv2.drawContours(contour_vis, r_circ, -1, (0, 165, 255), 2)
                cv2.drawContours(contour_vis, r_aspect, -1, (0, 255, 255), 2)
                cv2.drawContours(contour_vis, accepted_candidates, -1, (0, 255, 0), 2)
                draw_roi(contour_vis, roi_slice, roi_offset, tracker.use_roi, tracker.consistent_detection_count)
                
                final = frame_proc.copy()
                draw_trail(final, tracker.trail)
                draw_roi(final, roi_slice, roi_offset, tracker.use_roi, tracker.consistent_detection_count)

                if best:
                    cx, cy, radius, _ = best
                    cv2.circle(final, (cx, cy), radius + 4, (0, 255, 0), 2)
                    cv2.circle(final, (cx, cy), 3, (0, 255, 0), -1)
                    cv2.putText(final, "DETECTED", (cx + radius + 5, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif tracker.initialized and tracker.predicted:
                    px, py = tracker.predicted
                    cv2.drawMarker(final, (px, py), (0, 165, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(final, f"PREDICTED (miss:{tracker.missed})",
                                (px + 12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                cv2.putText(final, f"Frame {frame_n}", (10, PROCESS_HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Add status text
                if tracker.consistent_detection_count >= MIN_CONSISTENT_DETECTIONS:
                    if tracker.use_roi:
                        status_text = f"ROI ACTIVE - {tracker.consistent_detection_count} consistent detections"
                        status_color = (0, 255, 0)
                    else:
                        status_text = f"ROI PRIORITY - {tracker.consistent_detection_count} consistent (waiting for ROI candidate)"
                        status_color = (0, 255, 255)
                else:
                    status_text = f"ROI INACTIVE - need {MIN_CONSISTENT_DETECTIONS - tracker.consistent_detection_count} more consistent detections"
                    status_color = (0, 0, 255)
                
                cv2.putText(final, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                cand_vis = make_candidate_vis(gray, candidates, tracker, roi_slice, roi_offset)

                quad = assemble_panels([
                    (frame_proc, "Original"),
                    (mask_clean, "Mask Cleaned"),
                    (contour_vis, "Contours (Green=Accepted)"),
                    (cand_vis, "Candidates (Green=ROI)"),
                    (final, "Tracked Output"),
                ])

                frame_stop = time.perf_counter_ns()
                elapsed_ms = (frame_stop - frame_start) / 1e6
                frame_rate = 1000 / elapsed_ms if elapsed_ms > 0 else float('inf')
                cv2.putText(quad, f"FPS: {frame_rate:.1f}", (DISPLAY_WIDTH - 120, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Shot Put Tracker", quad)
                out_video.write(quad)

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
                    print("Background model and tracker reset.")

    total_compute_end = time.perf_counter()
    total_compute_time = total_compute_end - total_compute_start

    if total_frames_processed > 0:
        avg_time_per_frame = (total_compute_time / total_frames_processed) * 1000
        effective_fps = total_frames_processed / total_compute_time
    else:
        avg_time_per_frame = 0
        effective_fps = 0

    print("\n===== COMPUTE SUMMARY =====")
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total compute time: {total_compute_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.2f} ms")
    print(f"Effective processing FPS: {effective_fps:.2f}")
    print("===========================\n")

    if render_visualization:
        out_video.release()

    cap.release()
    if render_visualization:
        cv2.destroyAllWindows()


def extract_trajectory_from_video(video_path: str, max_frames: Optional[int] = None):
    """
    Programmatic extractor that uses the ROI-priority pipeline in this module.
    Returns the same trajectory (red line) as shown in the visualization.
    
    Returns:
        positions: List of (x, y) pixel coordinates for every frame (None if no position)
        detected: List of booleans indicating if position came from detection (True) or prediction (False)
        fps: Video frame rate
        (PROCESS_WIDTH, PROCESS_HEIGHT): Processing dimensions
        trail: List of all positions used for the red line trail (same as positions but filtered)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))

    tracker = Tracker()
    trail = deque(maxlen=TRAIL_LENGTH)
    frame_n = 0

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_n >= max_frames:
            break

        frame_n += 1
        frame_proc = cv2.resize(frame_orig, (PROCESS_WIDTH, PROCESS_HEIGHT))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        roi_slice, roi_offset = get_roi_from_prediction(tracker, gray.shape)
        fg_mask = bg_sub.apply(gray)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
        candidates, _, _, _, _, _ = detect_candidates_full(mask_clean)

        if tracker.initialized:
            tracker.predict()
        best = pick_best_candidate_with_roi_priority(candidates, tracker, roi_slice, roi_offset)

        if best:
            cx, cy, _radius, _circ = best
            tracker.correct(cx, cy)
            tracker.missed = 0
            trail.append(np.array([float(cx), float(cy)], dtype=np.float64))
        else:
            tracker.missed += 1
            if tracker.missed > MAX_MISSED_FRAMES:
                tracker.reset()
                trail.clear()

    cap.release()
    
    positions = list(trail)
    detected = [True] * len(positions)
    
    return positions, detected, fps, (PROCESS_WIDTH, PROCESS_HEIGHT), positions


# Enhanced plotting function that exactly matches the visualization
def plot_trajectory_like_visualization(video_path, output_path="trajectory_plot.png"):
    """
    Plot the trajectory exactly as shown in the red line of the visualization.
    """
    # Extract trajectory including the trail
    positions, detected, fps, (width, height), trail = extract_trajectory_from_video(video_path)
    
    if not trail:
        print("No trajectory data found!")
        return
    
    trail_array = np.array(trail)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory in image coordinates (like the red line)
    ax1.plot(trail_array[:, 0], trail_array[:, 1], 'r-', linewidth=2.5, alpha=0.8, label='Trajectory (red line)')
    
    # Color code detection vs prediction
    detection_colors = []
    frame_idx = 0
    for i, pos in enumerate(positions):
        if pos is not None:
            if detected[i]:
                detection_colors.append(('green', frame_idx, 'Detected'))
            else:
                detection_colors.append(('orange', frame_idx, 'Predicted'))
            frame_idx += 1
    
    # Plot points with colors
    for color, idx, label in detection_colors:
        ax1.scatter(trail_array[idx, 0], trail_array[idx, 1], 
                   c=color, s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Mark start and end
    ax1.scatter(trail_array[0, 0], trail_array[0, 1], 
               c='blue', s=200, marker='*', label='Start', zorder=5)
    ax1.scatter(trail_array[-1, 0], trail_array[-1, 1], 
               c='red', s=200, marker='*', label='End', zorder=5)
    
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.set_title(f'Shot Put Trajectory (Red Line from Visualization)\nVideo: {Path(video_path).name}', fontsize=14)
    ax1.invert_yaxis()  # Match image coordinates
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add text box with statistics
    stats_text = f'Total frames: {len(positions)}\n'
    stats_text += f'Valid positions: {len(trail)}\n'
    stats_text += f'Detected: {sum(detected)}\n'
    stats_text += f'Predicted: {len([d for d in detected if d is False])}\n'
    stats_text += f'FPS: {fps:.1f}\n'
    stats_text += f'X range: {trail_array[:, 0].min():.1f} - {trail_array[:, 0].max():.1f}\n'
    stats_text += f'Y range: {trail_array[:, 1].min():.1f} - {trail_array[:, 1].max():.1f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Position over time (like tracking visualization)
    frames = range(len(trail))
    ax2.plot(frames, trail_array[:, 0], 'r-', label='X coordinate', alpha=0.7)
    ax2.plot(frames, trail_array[:, 1], 'b-', label='Y coordinate', alpha=0.7)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Pixel Coordinate', fontsize=12)
    ax2.set_title('Position vs Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Trajectory Statistics ===")
    print(f"Total frames processed: {len(positions)}")
    print(f"Frames with position (red line points): {len(trail)}")
    print(f"  - Detected positions (green): {sum(detected)}")
    print(f"  - Predicted positions (orange): {len([d for d in detected if d is False])}")
    print(f"Trajectory length: {len(trail)} points")
    print(f"Plot saved to: {output_path}")
    
    return trail_array, detected, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot put tracker with ROI priority")
    parser.add_argument("--video", required=True, help="Path to .mp4 video file")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable all rendering for performance benchmarking (deprecated, use --no-render)")
    parser.add_argument("--render", action="store_true", 
                        help="Enable visualization rendering (default: True)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visualization rendering")
    
    args = parser.parse_args()
    
    if args.no_render or args.no_display:
        render_visualization = False
    else:
        render_visualization = True
    
    main(args.video, render_visualization=render_visualization)