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

  kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
  kf.processNoiseCov[2, 2] = 1.0  # allow velocity to change (throw arc)
  kf.processNoiseCov[3, 3] = 1.0

  kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

  kf.errorCovPost = np.eye(4, dtype=np.float32)
  return kf

GRAVITY_PX_PER_FRAME2 = 0.6 # downward acceleration in process-space pixels (number out of my ass)

@dataclass
class Tracker:
  kf: cv2.KalmanFilter = field(default_factory=make_kalman)
  initialized: bool = False
  missed: int = 0
  trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
  predicted: Optional[tuple] = None

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
    self.trail.append((cx, cy))
    self.missed = 0

  def reset(self):
    self.__init__()

def detect_candidates(mask):
  """
  Find circular blobs in a binary mask that could be the shot put.
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
    area = cv2.contourArea(cnt) # purple
    if area < MIN_AREA or area > MAX_AREA:
      rejected_area.append(cnt)
      continue

    perimeter = cv2.arcLength(cnt, True) # red
    if perimeter == 0 or perimeter > MAX_PERIMETER:
      rejected_perimeter.append(cnt)
      continue

    circularity = (4 * np.pi * area) / (perimeter ** 2) # orange
    if circularity < MIN_CIRCULARITY:
      rejected_circularity.append(cnt)
      continue

    _, _, w, h = cv2.boundingRect(cnt) # yellow
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
      rejected_aspect.append(cnt)
      continue

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    candidates.append((int(cx), int(cy), int(radius), circularity))
    accepted.append(cnt)

  return candidates, accepted, rejected_area, rejected_perimeter, rejected_circularity, rejected_aspect

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
        return min(gated, key=lambda c: np.hypot(c[0] - px, c[1] - py))
      # If nothing within gate, fall back to all candidates
      # (handles the launch moment when first entering frame)

  # If no tracker yet pick most circular
  return max(candidates, key=lambda c: c[3]) if candidates else None

def draw_trail(frame, trail, color=(0, 255, 255)):
  pts = list(trail)
  for i in range(1, len(pts)):
    alpha = i / len(pts)
    c = tuple(int(v * alpha) for v in color)
    cv2.line(frame, pts[i-1], pts[i], c, 2)


def make_candidate_vis(frame_gray, candidates, tracker):
  """Coloured visualization of candidate blobs."""
  vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
  for cx, cy, r, circ in candidates:
    cv2.circle(vis, (cx, cy), r, (0, 200, 255), 2)
    cv2.putText(vis, f"{circ:.2f}", (cx - 15, cy - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
  if tracker.predicted:
    px, py = tracker.predicted
    cv2.drawMarker(vis, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(vis, (px, py), 80, (255, 0, 255), 1) # gate circle
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

def main(video_path: str):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps = cap.get(cv2.CAP_PROP_FPS)
  print(f"Video: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x" f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {fps:.1f}fps")

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

  cv2.namedWindow("Shot Put Tracker", cv2.WINDOW_NORMAL)
  cv2.resizeWindow("Shot Put Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)

  while True:
    if not paused:
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

      fg_mask = bg_sub.apply(gray)
      bg_sub_end = time.perf_counter_ns()

      # Threshold to binary (MOG2 returns 127 for shadows if enabled,
      # 255 for foreground; since shadows off, just threshold at 200)
      _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
      mask_end = time.perf_counter_ns()

      mask_clean_inter = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)
      mask_clean = cv2.morphologyEx(mask_clean_inter, cv2.MORPH_CLOSE, kernel_close)
      morph_end = time.perf_counter_ns()

      candidates, accepted_candidates, r_area, r_per, r_circ, r_aspect = detect_candidates(mask_clean)
      canidate_detect_end = time.perf_counter_ns()
      contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
      cv2.drawContours(contour_vis, r_area, -1, (255, 0, 255), 2) # purple
      cv2.drawContours(contour_vis, r_per, -1, (0, 0, 255), 2) # red
      cv2.drawContours(contour_vis, r_circ, -1, (0, 165, 255), 2) # orange
      cv2.drawContours(contour_vis, r_aspect, -1, (0, 255, 255), 2) # yellow
      cv2.drawContours(contour_vis, accepted_candidates, -1, (0, 255, 0), 2) 
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
        print(f"Frame {frame_n}: read {(frame_read_end - frame_start) / 1e6:.1f}ms, downsample {(frame_downsample_end - frame_read_end) / 1e6:.1f}ms, bg_sub {(bg_sub_end - frame_downsample_end) / 1e6:.1f}ms, mask {(mask_end - bg_sub_end) / 1e6:.1f}ms, morph {(morph_end - mask_end) / 1e6:.1f}ms, candidate_detect {(canidate_detect_end - morph_end) / 1e6:.1f}ms, predict {(tracker_predict_end - tracker_predict_start) / 1e6:.1f}ms, select {(candidate_selection_end - candidate_select_start) / 1e6:.1f}ms")
      
      frame_stop = time.perf_counter_ns()
      elapsed_ms = (frame_stop - frame_start) / 1e6
      frame_rate = 1000 / elapsed_ms if elapsed_ms > 0 else float('inf')

      final = frame_proc.copy()
      draw_trail(final, tracker.trail)

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

      cand_vis = make_candidate_vis(gray, candidates, tracker)

      quad = assemble_panels([
        (frame_proc, "Original"),
        (mask_clean, "Mask Cleaned"),
        (contour_vis, "Contours (Green=Accepted, Red=Rejected)"),
        (cand_vis, "Candidates"),
        (final, "Tracked Output"),
      ])

      cv2.putText(quad, f"FPS: {frame_rate:.1f}", (DISPLAY_WIDTH - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Shot Put Tracker", quad)

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

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Shot put tracker with visualization")
  parser.add_argument("--video", required=True, help="Path to .mp4 video file")
  args = parser.parse_args()
  main(args.video)
