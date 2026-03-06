"""
Shot Put Tracker

Usage:
  python shotput_tracker.py --video path/to/video.mp4

Controls:
  SPACE - pause/resume
  Q     - quit
  R     - reset tracker
Red → Moving right (0 degrees)

Green → Moving up (90 degrees)

Blue → Moving left (180 degrees)

Purple/Pink → Moving down (270 degrees)

Yellow → Diagonal up-right (45 degrees)

Cyan → Diagonal up-left (135 degrees)
inform selection based on hue? 
"""


MORPH_OPEN_KERNEL  = 3   # removes small noise blobs
MORPH_CLOSE_KERNEL = 40  # fills holes inside the shot put blob

import cv2
import numpy as np
import argparse
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
import argparse
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Optional, Tuple, List

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540

# Optical flow parameters
OPTICAL_FLOW_SCALE = 0.5  # Scale down further for flow calculation
MOTION_THRESHOLD = 15.0  # Lower threshold for motion sensitivity
MIN_FLOW_POINTS = 5

# ROI tracking after initialization
ROI_SIZE = 50  # pixels in process-space (width and height of ROI)
ROI_PADDING = 20  # extra padding around predicted position
MIN_ROI_SIZE = 100  # minimum ROI size when not initialized

# Consistency check parameters
CONSISTENCY_WINDOW = 5
MAX_DISTANCE_VARIATION = 30
MIN_CONSISTENT_DETECTIONS = 3

# Detection parameters
MIN_AREA = 8
MAX_AREA = 200
MAX_PERIMETER = 70
MIN_CIRCULARITY = 0.45
MAX_ASPECT_RATIO = 1.7

MAX_MISSED_FRAMES = 8
TRAIL_LENGTH = 60

def make_kalman():
    """State: [x, y, vx, vy] with gravity bias."""
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
    kf.processNoiseCov[2, 2] = 1.0
    kf.processNoiseCov[3, 3] = 1.0

    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
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
    last_position: Optional[tuple] = None
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=CONSISTENCY_WINDOW))
    use_roi: bool = False

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
            return

        positions = list(self.recent_positions)
        distances = []
        for i in range(1, len(positions)):
            dist = np.hypot(positions[i][0] - positions[i-1][0],
                           positions[i][1] - positions[i-1][1])
            distances.append(dist)

        if len(distances) < 2:
            self.use_roi = False
            return

        max_dist = max(distances)
        min_dist = min(distances)

        if (max_dist - min_dist) < MAX_DISTANCE_VARIATION and max_dist < MAX_DISTANCE_VARIATION * 2:
            self.use_roi = True
        else:
            self.use_roi = False

    def reset(self):
        self.__init__()


class OpticalFlowMotionDetector:
    """Detects motion using optical flow between frames."""
    
    def __init__(self):
        self.prev_gray = None
        self.first_frame = True
        self.debug_flow = None
        
    def detect_motion(self, frame_gray):
        """
        Detect motion using optical flow.
        Returns a binary motion mask and flow magnitude visualization.
        """
        h, w = frame_gray.shape
        
        # Scale down for faster flow calculation
        small_gray = cv2.resize(frame_gray, (0, 0), fx=OPTICAL_FLOW_SCALE, fy=OPTICAL_FLOW_SCALE)
        small_h, small_w = small_gray.shape
        
        if self.first_frame:
            self.prev_gray = small_gray.copy()
            self.first_frame = False
            # Return blank mask for first frame
            return np.zeros_like(frame_gray), np.zeros_like(frame_gray)
        
        try:
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, small_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            # Calculate flow magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Normalize magnitude for visualization (0-255)
            mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Create HSV visualization of flow
            hsv = np.zeros((small_h, small_w, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
            hsv[..., 1] = 255  # Saturation = max
            hsv[..., 2] = mag_norm  # Value = magnitude
            
            # Convert HSV to BGR for visualization
            flow_vis_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_vis = cv2.resize(flow_vis_small, (w, h))
            
            # Create binary motion mask where magnitude exceeds threshold
            motion_mask_small = (mag > MOTION_THRESHOLD).astype(np.uint8) * 255
            motion_mask = cv2.resize(motion_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Update previous frame
            self.prev_gray = small_gray.copy()
            
            return motion_mask, flow_vis
            
        except Exception as e:
            print(f"Optical flow error: {e}")
            return np.zeros_like(frame_gray), np.zeros_like(frame_gray)
    
    def reset(self):
        self.first_frame = True
        self.prev_gray = None


def get_roi_from_prediction(tracker: Tracker, frame_shape: Tuple[int, int]) -> Tuple[slice, slice, Tuple[int, int]]:
    """Get ROI slices based on tracker prediction."""
    h, w = frame_shape[:2]

    if tracker.initialized and tracker.predicted and tracker.use_roi:
        cx, cy = tracker.predicted
        if len(tracker.trail) > 2:
            vx = tracker.kf.statePost[2][0]
            vy = tracker.kf.statePost[3][0]
            speed = np.hypot(vx, vy)
            dynamic_roi = int(ROI_SIZE * (1.0 + speed / 20.0))
            roi_size = min(ROI_SIZE * 2, max(ROI_SIZE, dynamic_roi))
        else:
            roi_size = ROI_SIZE
    else:
        cx, cy = w // 2, h // 2
        roi_size = MIN_ROI_SIZE

    left = max(0, cx - roi_size // 2 - ROI_PADDING)
    right = min(w, cx + roi_size // 2 + ROI_PADDING)
    top = max(0, cy - roi_size // 2 - ROI_PADDING)
    bottom = min(h, cy + roi_size // 2 + ROI_PADDING)

    return (slice(top, bottom), slice(left, right)), (top, left)


def detect_candidates_in_roi(mask, roi_slice, roi_offset):
    """Find circular blobs in ROI. Returns candidates in full-frame coordinates."""
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
    """Find circular blobs in full frame."""
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


def pick_best_candidate(candidates, tracker: Tracker):
    """Pick the best candidate based on score."""
    if not candidates:
        return None

    if tracker.initialized:
        vx = tracker.kf.statePost[2][0]
        if vx > 2.0 and tracker.predicted:
            filtered = [c for c in candidates if c[0] >= tracker.predicted[0] - 10]
            if filtered:
                candidates = filtered

        if tracker.predicted:
            px, py = tracker.predicted
            gate = 80
            gated = [c for c in candidates if np.hypot(c[0] - px, c[1] - py) < gate]

            if gated and len(tracker.trail) > 2:
                avg_speed = np.mean([
                    np.hypot(tracker.trail[i][0] - tracker.trail[i-1][0],
                            tracker.trail[i][1] - tracker.trail[i-1][1])
                    for i in range(1, len(tracker.trail))
                ])
                max_allowed = avg_speed * 2.5
                speed_gated = [c for c in gated if np.hypot(c[0] - px, c[1] - py) <= max_allowed]
                if speed_gated:
                    gated = speed_gated
            if gated:
                return min(gated, key=lambda c: candidate_score(c, tracker))

    return max(candidates, key=lambda c: c[3]) if candidates else None


def draw_trail(frame, trail):
    """Draw the trajectory trail."""
    pts = list(trail)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 3)


def draw_roi(frame, roi_slice, roi_offset, use_roi=True):
    """Draw ROI rectangle with status color."""
    top, left = roi_offset
    bottom = top + (roi_slice[0].stop - roi_slice[0].start)
    right = left + (roi_slice[1].stop - roi_slice[1].start)

    if use_roi:
        color = (0, 255, 0)
        label = "ROI (ACTIVE)"
    else:
        color = (0, 0, 255)
        label = "ROI (INACTIVE)"

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, label, (left + 5, top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def make_candidate_vis(frame_gray, candidates, tracker, roi_slice=None, roi_offset=None, motion_mask=None):
    """Visualize candidates and tracking state."""
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Overlay motion mask as semi-transparent red
    if motion_mask is not None and motion_mask.any():
        # Create red overlay for motion regions
        motion_overlay = np.zeros_like(vis)
        motion_overlay[:, :, 2] = motion_mask  # Red channel for motion
        vis = cv2.addWeighted(vis, 0.7, motion_overlay, 0.3, 0)
        
        # Add text showing motion detected
        motion_pixels = np.count_nonzero(motion_mask)
        if motion_pixels > 100:  # Only show if significant motion
            cv2.putText(vis, f"Motion: {motion_pixels}px", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if roi_slice is not None and roi_offset is not None:
        top, left = roi_offset
        bottom = top + (roi_slice[0].stop - roi_slice[0].start)
        right = left + (roi_slice[1].stop - roi_slice[1].start)

        if tracker.use_roi:
            color = (0, 255, 0)
            label = "ROI (ACTIVE)"
        else:
            color = (0, 0, 255)
            label = "ROI (INACTIVE)"

        cv2.rectangle(vis, (left, top), (right, bottom), color, 2)
        cv2.putText(vis, label, (left + 5, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for cx, cy, r, circ in candidates:
        cv2.circle(vis, (cx, cy), r, (0, 200, 255), 2)
        cv2.putText(vis, f"{circ:.2f}", (cx - 15, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    if tracker.predicted:
        px, py = tracker.predicted
        cv2.drawMarker(vis, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, (px, py), 80, (255, 0, 255), 1)

    # Add consistency status
    if len(tracker.recent_positions) >= MIN_CONSISTENT_DETECTIONS:
        positions = list(tracker.recent_positions)
        distances = []
        for i in range(1, len(positions)):
            dist = np.hypot(positions[i][0] - positions[i-1][0],
                           positions[i][1] - positions[i-1][1])
            distances.append(dist)

        if distances:
            max_dist = max(distances)
            min_dist = min(distances)
            variation = max_dist - min_dist
            status_color = (0, 255, 0) if variation < MAX_DISTANCE_VARIATION else (0, 0, 255)
            cv2.putText(vis, f"Dist var: {variation:.1f}px", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    return vis


def assemble_panels(panels: list[tuple[np.ndarray, str]]):
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
        r = resized_with_labels[i + 1][0] if i + 1 < len(resized_with_labels) else np.zeros_like(l)
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
    print("Using Optical Flow for motion detection")

    # Initialize optical flow motion detector
    motion_detector = OpticalFlowMotionDetector()

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))

    tracker = Tracker()
    paused = False
    frame_n = 0

    if render_visualization:
        cv2.namedWindow("Shot Put Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Shot Put Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    total_compute_start = time.perf_counter()
    total_frames_processed = 0

    while True:
        if not paused:
            frame_start = time.perf_counter_ns()
            ret, frame_orig = cap.read()
            frame_read_end = time.perf_counter_ns()
            if not ret:
                print("End of video.")
                break
            frame_n += 1

            # Downsample for processing
            frame_proc = cv2.resize(frame_orig, (PROCESS_WIDTH, PROCESS_HEIGHT))
            gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
            frame_downsample_end = time.perf_counter_ns()

            # Get ROI based on tracker prediction
            roi_slice, roi_offset = get_roi_from_prediction(tracker, gray.shape)

            # Detect motion using optical flow
            motion_mask, flow_vis = motion_detector.detect_motion(gray)
            flow_detect_end = time.perf_counter_ns()

            # Apply morphological operations to clean up motion mask
            if motion_mask.any():
                mask_clean_inter = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_open)
                mask_clean = cv2.morphologyEx(mask_clean_inter, cv2.MORPH_CLOSE, kernel_close)
            else:
                mask_clean = motion_mask
            morph_end = time.perf_counter_ns()

            # Detect candidates - either in ROI or full frame based on consistency
            if tracker.initialized and tracker.use_roi:
                candidates, accepted_candidates, r_area, r_per, r_circ, r_aspect = detect_candidates_in_roi(
                    mask_clean, roi_slice, roi_offset
                )
            else:
                candidates, accepted_candidates, r_area, r_per, r_circ, r_aspect = detect_candidates_full(
                    mask_clean
                )
            candidate_detect_end = time.perf_counter_ns()

            # Visualization
            if render_visualization:
                contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_vis, r_area, -1, (255, 0, 255), 2)
                cv2.drawContours(contour_vis, r_per, -1, (0, 0, 255), 2)
                cv2.drawContours(contour_vis, r_circ, -1, (0, 165, 255), 2)
                cv2.drawContours(contour_vis, r_aspect, -1, (0, 255, 255), 2)
                cv2.drawContours(contour_vis, accepted_candidates, -1, (0, 255, 0), 2)

                draw_roi(contour_vis, roi_slice, roi_offset, use_roi=tracker.use_roi)

            # Kalman prediction
            tracker_predict_start = time.perf_counter_ns()
            if tracker.initialized:
                tracker.predict()
            tracker_predict_end = time.perf_counter_ns()

            # Select best candidate
            candidate_select_start = time.perf_counter_ns()
            best = pick_best_candidate(candidates, tracker)
            if best:
                cx, cy, radius, _ = best
                tracker.correct(cx, cy)
                tracker.missed = 0
            else:
                tracker.missed += 1
                if tracker.missed > MAX_MISSED_FRAMES:
                    tracker.reset()
                    motion_detector.reset()  # Reset optical flow on tracker reset
            candidate_selection_end = time.perf_counter_ns()

            # Periodic timing output
            if frame_n % 15 == 0:
                motion_pixels = np.count_nonzero(motion_mask) if motion_mask is not None else 0
                print(f"Frame {frame_n}: read {(frame_read_end - frame_start) / 1e6:.1f}ms, "
                      f"downsample {(frame_downsample_end - frame_read_end) / 1e6:.1f}ms, "
                      f"optical_flow {(flow_detect_end - frame_downsample_end) / 1e6:.1f}ms, "
                      f"morph {(morph_end - flow_detect_end) / 1e6:.1f}ms, "
                      f"candidate_detect {(candidate_detect_end - morph_end) / 1e6:.1f}ms, "
                      f"predict {(tracker_predict_end - tracker_predict_start) / 1e6:.1f}ms, "
                      f"select {(candidate_selection_end - candidate_select_start) / 1e6:.1f}ms, "
                      f"use_roi={tracker.use_roi}, motion_pixels={motion_pixels}")

            # Calculate FPS
            frame_stop = time.perf_counter_ns()
            elapsed_ms = (frame_stop - frame_start) / 1e6
            frame_rate = 1000 / elapsed_ms if elapsed_ms > 0 else float('inf')

            total_frames_processed += 1

            # Visualization
            if render_visualization:
                final = frame_proc.copy()
                draw_trail(final, tracker.trail)
                draw_roi(final, roi_slice, roi_offset, use_roi=tracker.use_roi)

                if best:
                    cx, cy, radius, _ = best
                    cv2.circle(final, (cx, cy), radius + 4, (0, 255, 0), 2)
                    cv2.circle(final, (cx, cy), 3, (0, 255, 0), -1)
                    cv2.putText(final, "DETECTED", (cx + radius + 5, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif tracker.initialized and tracker.predicted:
                    px, py = tracker.predicted
                    cv2.drawMarker(final, (px, py), (0, 165, 255),
                                    cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(final, f"PREDICTED (miss:{tracker.missed})",
                                (px + 12, py), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 165, 255), 1)

                cv2.putText(final, f"Frame {frame_n}", (10, PROCESS_HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                status_text = "ROI ACTIVE" if tracker.use_roi else "ROI INACTIVE"
                status_color = (0, 255, 0) if tracker.use_roi else (0, 0, 255)
                cv2.putText(final, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                cand_vis = make_candidate_vis(gray, candidates, tracker, roi_slice, roi_offset, motion_mask)

                # Prepare panels - use flow visualization instead of blank mask
                if flow_vis is not None and flow_vis.any():
                    flow_panel = flow_vis
                else:
                    flow_panel = np.zeros_like(frame_proc)

                quad = assemble_panels([
                    (frame_proc, "Original"),
                    (flow_panel, "Optical Flow"),
                    (contour_vis, "Contours"),
                    (cand_vis, "Candidates"),
                    (final, "Tracked Output"),
                ])

                cv2.putText(quad, f"FPS: {frame_rate:.1f}", (DISPLAY_WIDTH - 120, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("Shot Put Tracker", quad)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r'):
                    tracker.reset()
                    motion_detector.reset()
                    print("Tracker and motion detector reset.")
        else:
            if render_visualization:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("Resumed")
                elif key == ord('r'):
                    tracker.reset()
                    motion_detector.reset()
                    print("Tracker and motion detector reset.")
            else:
                time.sleep(0.1)

    # Performance summary
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

    cap.release()
    if render_visualization:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot put tracker with optical flow")
    parser.add_argument("--video", required=True, help="Path to .mp4 video file")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable all rendering for performance benchmarking")
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