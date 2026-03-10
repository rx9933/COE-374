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
from typing import Optional, Tuple, List

DISPLAY_WIDTH  = 1920
DISPLAY_HEIGHT = 1080
PROCESS_WIDTH  = 960
PROCESS_HEIGHT = 540

# Background subtractor
MOG2_HISTORY        = 300   # frames to build background model
MOG2_VAR_THRESHOLD  = 80    # lower = more sensitive; raise if noisy background
MOG2_DETECT_SHADOWS = False  # TODO: try with this true...

MORPH_OPEN_KERNEL  = 3   # removes small noise blobs
MORPH_CLOSE_KERNEL = 40  # fills holes inside the shot put blob

# ROI tracking after initialization
ROI_SIZE = 100  # pixels in process-space (width and height of ROI)
ROI_PADDING = 20  # extra padding around predicted position
MIN_ROI_SIZE = 100  # minimum ROI size when not initialized

# Consistency check parameters
CONSISTENCY_WINDOW = 5  # number of frames to check for consistency
MAX_DISTANCE_VARIATION = 30  # maximum allowed variation in distances (pixels)
MIN_CONSISTENT_DETECTIONS = 3  # minimum number of detections needed in window

MIN_AREA            = 8    # px^2 — ignore tiny noise
MAX_AREA            = 200   # px^2 — ignore huge regions
MAX_PERIMETER       = 70    # px — ignore very large contours (athlete body)
MIN_CIRCULARITY     = 0.45   # 1.0 = perfect circle; lower catches slight blur
MAX_ASPECT_RATIO    = 1.7    # width/height of bounding rect; rejects lines

"""

Note - the rolling shutter of the GoPro really fucks us
If we used a camera with global shutter
We could have much stricter circularity and aspect ratio thresholds and way fewer false positives
This is in reference to parameters above this comment

"""

MAX_MISSED_FRAMES   = 8     # frames without detection before tracker resets
TRAIL_LENGTH        = 60    # how many past positions to draw as trail

def make_kalman():
  """
  State:  [x, y, vx, vy]
  Measurement: [x, y]
  We inject a small gravity bias into the prediction step manually.
  """
  kf = cv2.KalmanFilter(4, 2)

  # Transition matrix (constant velocity)
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

  kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-7#2
  kf.processNoiseCov[2, 2] = 1.0  # allow velocity to change (throw arc)
  kf.processNoiseCov[3, 3] = 1.0

  kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0 

  kf.errorCovPost = np.eye(4, dtype=np.float32)
  return kf

GRAVITY_PX_PER_FRAME2 = 0.6# downward acceleration in process-space pixels (number out of my ass)

@dataclass
class Tracker:
  kf: cv2.KalmanFilter = field(default_factory=make_kalman)
  initialized: bool = False
  missed: int = 0
  trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
  predicted: Optional[tuple] = None
  last_position: Optional[tuple] = None
  # Store recent positions for consistency check
  recent_positions: deque = field(default_factory=lambda: deque(maxlen=CONSISTENCY_WINDOW))
  # Flag to indicate if we should use ROI
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
      # Seed velocity with zeros on first detection
      self.kf.statePost[0] = cx
      self.kf.statePost[1] = cy
      self.initialized = True
    
    self.last_position = (cx, cy)
    self.trail.append((cx, cy))
    self.recent_positions.append((cx, cy))
    self.missed = 0
    
    # Update ROI usage flag based on consistency check
    self.update_roi_flag()

  def update_roi_flag(self):
    """Check if recent positions are consistent enough to use ROI."""
    if len(self.recent_positions) < MIN_CONSISTENT_DETECTIONS:
      self.use_roi = False
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
    mean_dist = np.mean(distances)
    max_dist = max(distances)
    min_dist = min(distances)
    
    # Variation should be within threshold
    # Also check that we're not seeing huge jumps (exploding identifications)
    if (max_dist - min_dist) < MAX_DISTANCE_VARIATION and max_dist < MAX_DISTANCE_VARIATION * 2:
      self.use_roi = True
    else:
      self.use_roi = False

  def reset(self):
    self.__init__()

def get_roi_from_prediction(tracker: Tracker, frame_shape: Tuple[int, int]) -> Tuple[slice, slice, Tuple[int, int]]:
    """
    Get ROI slices based on tracker prediction.
    Returns (y_slice, x_slice, offset) where offset is (top, left) for mapping back to full frame.
    """
    h, w = frame_shape[:2]
    
    if tracker.initialized and tracker.predicted and tracker.use_roi:
        # Use predicted position as ROI center
        cx, cy = tracker.predicted
        # Dynamically adjust ROI size based on velocity
        if len(tracker.trail) > 2:
            vx = tracker.kf.statePost[2][0]
            vy = tracker.kf.statePost[3][0]
            speed = np.hypot(vx, vy)
            # Larger ROI for faster movement
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

def detect_candidates_in_roi(mask, roi_slice, roi_offset):
    """
    Find circular blobs in a binary mask within the specified ROI.
    Returns list of (cx, cy, radius, circularity) tuples in full-frame coordinates.
    """
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

        # Get center in ROI coordinates, then convert to full-frame
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
    """
    Find circular blobs in a binary mask across the full frame.
    Returns list of (cx, cy, radius, circularity) tuples.
    """
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

        # Get center
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            candidates.append((int(cx), int(cy), int(radius), circularity))
            accepted.append(cnt)

    return candidates, accepted, rejected_area, rejected_perimeter, rejected_circularity, rejected_aspect

def candidate_score(candidate, tracker: Tracker):
  """
  Lower score = better candidate.
  Combines distance, velocity consistency, size stability, and circularity.
  """

  cx, cy, r, circ = candidate

  if tracker.predicted is None:
    return -circ  # prefer most circular if no prediction

  px, py = tracker.predicted

  # distance from predicted location
  dist = np.hypot(cx - px, cy - py)

  # velocity consistency
  vx = tracker.kf.statePost[2][0]
  vy = tracker.kf.statePost[3][0]

  expected_x = px + vx
  expected_y = py + vy

  vel_err = np.hypot(cx - expected_x, cy - expected_y)

  # radius stability
  if len(tracker.trail) > 0:
    prev_x, prev_y = tracker.trail[-1]
    size_err = abs(r - 6)  # FIX THIS
  else:
    size_err = 0
#   print("score",dist,vel_err, size_err, circ)
  score = (
      dist
      + 0.5 * vel_err
      + .5 * size_err
      - 2 * circ
  )

  return score

def pick_best_candidate(candidates, tracker: Tracker):
  """
  If the tracker has a predicted position, pick the candidate closest to it.
  Otherwise pick the most circular candidate.
  """
  if not candidates:
    return None
  
  if tracker.initialized:
    vx = tracker.kf.statePost[2][0]
    if vx > 2.0:
      filtered = [c for c in candidates if c[0] >= tracker.predicted[0] - 10]
      if filtered:
        candidates = filtered

    if tracker.predicted:
      px, py = tracker.predicted
      # Gate: only accept candidates within a reasonable search radius
      # faster-moving = larger gate
      gate = 80 # pixels in process-space
      gated = [c for c in candidates if np.hypot(c[0] - px, c[1] - py) < gate]

      if gated and len(tracker.trail) > 2:

        avg_speed = np.mean([
          np.hypot(tracker.trail[i][0]-tracker.trail[i - 1][0], tracker.trail[i][1]-tracker.trail[i - 1][1])
          for i in range(1, len(tracker.trail))
        ])
        max_allowed = avg_speed * 2.5
        speed_gated = [c for c in gated if np.hypot(c[0] - px, c[1] - py) <= max_allowed]
        if speed_gated:
          gated = speed_gated
          # if speed_gated is empty, keep original gated
      if gated:
        return min(gated, key=lambda c: candidate_score(c, tracker))
      
      # If nothing within gate, fall back to all candidates
      # (handles the launch moment when first entering frame)

  # If no tracker yet pick most circular
  return max(candidates, key=lambda c: c[3]) if candidates else None

def draw_trail(frame, trail, color=(0, 255, 255)):
  pts = list(trail)
  for i in range(1, len(pts)):
    # alpha = i / len(pts)
    # c = tuple(int(v * alpha) for v in color)
    # cv2.line(frame, pts[i-1], pts[i], c, 2)
    cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 3) 

def draw_roi(frame, roi_slice, roi_offset, color=(255, 0, 255), use_roi=True):
    """Draw ROI rectangle on frame with different color based on usage."""
    top, left = roi_offset
    bottom = top + (roi_slice[0].stop - roi_slice[0].start)
    right = left + (roi_slice[1].stop - roi_slice[1].start)
    
    # Use different color based on whether ROI is actually being used
    if use_roi:
        color = (0, 255, 0)  # Green when using ROI
        label = "ROI (ACTIVE)"
    else:
        color = (0, 0, 255)  # Red when not using ROI
        label = "ROI (INACTIVE - FULL SEARCH)"
    
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, label, (left + 5, top + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def make_candidate_vis(frame_gray, candidates, tracker, roi_slice=None, roi_offset=None):
  """Coloured visualization of candidate blobs."""
  vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
  
  # Draw ROI if provided
  if roi_slice is not None and roi_offset is not None:
      top, left = roi_offset
      bottom = top + (roi_slice[0].stop - roi_slice[0].start)
      right = left + (roi_slice[1].stop - roi_slice[1].start)
      
      # Use different color based on whether ROI is being used
      if tracker.use_roi:
          color = (0, 255, 0)  # Green when using ROI
          label = "ROI (ACTIVE)"
      else:
          color = (0, 0, 255)  # Red when not using ROI
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
    cv2.circle(vis, (px, py), 80, (255, 0, 255), 1) # gate circle
  
  # Add consistency status
  status_y = 50
  if len(tracker.recent_positions) >= MIN_CONSISTENT_DETECTIONS:
    positions = list(tracker.recent_positions)
    distances = []
    for i in range(1, len(positions)):
      dist = np.hypot(positions[i][0] - positions[i-1][0], 
                     positions[i][1] - positions[i-1][1])
      distances.append(dist)
    
    if distances:
      mean_dist = np.mean(distances)
      max_dist = max(distances)
      min_dist = min(distances)
      variation = max_dist - min_dist
      
      status_color = (0, 255, 0) if variation < MAX_DISTANCE_VARIATION else (0, 0, 255)
      cv2.putText(vis, f"Dist var: {variation:.1f}px", (10, status_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
  
  return vis

def assemble_panels(panels: list[tuple[np.ndarray, str]]):
  """Tile frames into one display."""
  num_rows = (len(panels) + 1) // 2
  target_h = DISPLAY_HEIGHT // num_rows
  target_w = DISPLAY_WIDTH if len(panels) == 1 else DISPLAY_WIDTH // 2

  def resize(f, target_w, target_h):
    h, w = f.shape[:2]
    scale = min(target_w / w, target_h / h) # preserve aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(f, (new_w, new_h))

    # pad to fill the target width/height
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



def extract_trajectory_from_video(video_path: str, max_frames: Optional[int] = None):
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
  positions = []
  detected = []

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
    fg_mask = bg_sub.apply(gray)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    candidates, _, _, _, _, _ = detect_candidates(mask_clean)


    if tracker.initialized:
      tracker.predict()
    best = pick_best_candidate(candidates, tracker)

    if best:
      cx, cy, radius, _ = best
      tracker.correct(cx, cy)
      tracker.missed = 0
      positions.append(np.array([float(cx), float(cy)], dtype=np.float64))
      detected.append(True)
    else:
      tracker.missed += 1
      if tracker.missed > MAX_MISSED_FRAMES:
        tracker.reset()
      if tracker.predicted is not None:
        positions.append(np.array([float(tracker.predicted[0]), float(tracker.predicted[1])], dtype=np.float64))
        detected.append(False)
      else:
        positions.append(None)
        detected.append(False)

  cap.release()
  return positions, detected, fps, (PROCESS_WIDTH, PROCESS_HEIGHT)

def main(video_path: str, render_visualization: bool = True):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps = cap.get(cv2.CAP_PROP_FPS)
  print(f"Video: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x" f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {fps:.1f}fps")
  print(f"Visualization: {'ON' if render_visualization else 'OFF'}")

  bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=MOG2_HISTORY,
    varThreshold=MOG2_VAR_THRESHOLD,
    detectShadows=MOG2_DETECT_SHADOWS,
  )

  kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
  kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))

  tracker = Tracker()
  paused  = False
  frame_n = 0

  if render_visualization:
    cv2.namedWindow("Shot Put Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shot Put Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('roi_kalm.mp4', fourcc, 30.0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

  total_compute_start = time.perf_counter()
  total_frames_processed = 0
  
  while True:
    if not paused:
      frame_compute_start = time.perf_counter()
      frame_start = time.perf_counter_ns()
      ret, frame_orig = cap.read()
      frame_read_end = time.perf_counter_ns()
      if not ret:
        print("End of video.")
        break
      frame_n += 1

      # Simplify for speed
      frame_proc = cv2.resize(frame_orig, (PROCESS_WIDTH, PROCESS_HEIGHT))
      gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
      frame_downsample_end = time.perf_counter_ns()

      # Get ROI based on tracker prediction
      roi_slice, roi_offset = get_roi_from_prediction(tracker, gray.shape)
      
      # Apply background subtraction - either full frame or ROI based on consistency
      if tracker.initialized and tracker.use_roi:
          # Only process ROI when consistent
          roi_mask = np.zeros_like(gray)
          roi_mask[roi_slice] = gray[roi_slice]
          fg_mask_full = bg_sub.apply(roi_mask)
          fg_mask = np.zeros_like(gray)
          fg_mask[roi_slice] = fg_mask_full[roi_slice]
      else:
          # Full frame processing when not consistent or not initialized
          fg_mask = bg_sub.apply(gray)
      
      bg_sub_end = time.perf_counter_ns()

      # Threshold to binary
      _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
      mask_end = time.perf_counter_ns()

      # Apply morphological operations
      mask_clean_inter = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
      mask_clean = cv2.morphologyEx(mask_clean_inter, cv2.MORPH_CLOSE, kernel_close)
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
      canidate_detect_end = time.perf_counter_ns()
        
      # Only create visualization frames if rendering is enabled
      if render_visualization:
        contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, r_area, -1, (255, 0, 255), 2) # purple
        cv2.drawContours(contour_vis, r_per, -1, (0, 0, 255), 2) # red
        cv2.drawContours(contour_vis, r_circ, -1, (0, 165, 255), 2) # orange
        cv2.drawContours(contour_vis, r_aspect, -1, (0, 255, 255), 2) # yellow
        cv2.drawContours(contour_vis, accepted_candidates, -1, (0, 255, 0), 2)
        
        # Draw ROI on visualization (will show active/inactive status)
        draw_roi(contour_vis, roi_slice, roi_offset, use_roi=tracker.use_roi)
      
      tracker_predict_start = time.perf_counter_ns()

      if tracker.initialized:
        tracker.predict()
      tracker_predict_end = time.perf_counter_ns()

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
      candidate_selection_end = time.perf_counter_ns()
      
      if frame_n % 15 == 0:
        print(f"Frame {frame_n}: read {(frame_read_end - frame_start) / 1e6:.1f}ms, downsample {(frame_downsample_end - frame_read_end) / 1e6:.1f}ms, bg_sub {(bg_sub_end - frame_downsample_end) / 1e6:.1f}ms, mask {(mask_end - bg_sub_end) / 1e6:.1f}ms, morph {(morph_end - mask_end) / 1e6:.1f}ms, candidate_detect {(canidate_detect_end - morph_end) / 1e6:.1f}ms, predict {(tracker_predict_end - tracker_predict_start) / 1e6:.1f}ms, select {(candidate_selection_end - candidate_select_start) / 1e6:.1f}ms, use_roi={tracker.use_roi}")
      
      frame_stop = time.perf_counter_ns()
      elapsed_ms = (frame_stop - frame_start) / 1e6
      frame_rate = 1000 / elapsed_ms if elapsed_ms > 0 else float('inf')

      # Only create visualization frames if rendering is enabled
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
        
        # Add ROI status text
        status_text = "ROI ACTIVE" if tracker.use_roi else "ROI INACTIVE (full frame search)"
        status_color = (0, 255, 0) if tracker.use_roi else (0, 0, 255)
        cv2.putText(final, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cand_vis = make_candidate_vis(gray, candidates, tracker, roi_slice, roi_offset)

        quad = assemble_panels([
          (frame_proc, "Original"),
          (mask_clean, "Mask Cleaned"),
          (contour_vis, "Contours (Green=Accepted, Red=Rejected)"),
          (cand_vis, "Candidates"),
          (final, "Tracked Output"),
        ])

        cv2.putText(quad, f"FPS: {frame_rate:.1f}", (DISPLAY_WIDTH - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
      
      frame_compute_end = time.perf_counter()
      total_frames_processed += 1
      
      if render_visualization:
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
    else:
      # If paused, just wait for key press
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
          
      else:
        # If no visualization and paused, just sleep a bit to avoid busy loop
        
        pass

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Shot put tracker with visualization")
  parser.add_argument("--video", required=True, help="Path to .mp4 video file")
  parser.add_argument("--no-display", action="store_true",
                      help="Disable all rendering for performance benchmarking (deprecated, use --no-render)")
  parser.add_argument("--render", action="store_true", 
                      help="Enable visualization rendering (default: True)")
  parser.add_argument("--no-render", action="store_true",
                      help="Disable visualization rendering")
  
  args = parser.parse_args()
  
  # Determine render flag:
  # Default is True
  # --no-render or --no-display will disable rendering
  # --render can explicitly enable rendering (though it's default)
  if args.no_render or args.no_display:
    render_visualization = False
  else:
    # If neither no-render nor no-display is set, default to True
    # But if --render is provided, it's still True (explicitly enabled)
    render_visualization = True
  
  main(args.video, render_visualization=render_visualization)