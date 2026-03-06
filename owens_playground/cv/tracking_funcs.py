
from params import *

import cv2
import numpy as np
import argparse
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Optional, Tuple, List
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


def draw_predictions(frame, tracker, best_candidate=None):
    """Draw bounding boxes and predictions on frame"""
    
    # Draw trail
    '''
    pts = list(tracker.trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in (0, 255, 255))
        cv2.line(frame, pts[i-1], pts[i], c, 2)
    '''
    # Draw trail - all in bright red
    pts = list(tracker.trail)
    for i in range(1, len(pts)):
        # Use bright red (0, 0, 255) for all segments
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 3)  # Bright red, thicker line

    # Draw prediction
    if tracker.initialized and tracker.predicted:
        px, py = tracker.predicted
        cv2.drawMarker(frame, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(frame, (px, py), 15, (255, 0, 255), 2)
        cv2.circle(frame, (px, py), 80, (255, 0, 255), 1)
        cv2.putText(frame, "PRED", (px + 20, py - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw detection
    if best_candidate:
        cx, cy, radius, circ = best_candidate
        cv2.circle(frame, (cx, cy), radius + 4, (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (cx - radius - 5, cy - radius - 5),
                     (cx + radius + 5, cy + radius + 5), (0, 255, 0), 2)
        cv2.putText(frame, f"DETECTED", (cx + radius + 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"circ:{circ:.2f}", (cx + radius + 10, cy + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw missed count
    if tracker.missed > 0 and tracker.predicted:
        px, py = tracker.predicted
        cv2.putText(frame, f"MISSED:{tracker.missed}", (px + 20, py + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # Draw status
    status = "TRACKING" if tracker.initialized else "SEARCHING"
    color = (0, 255, 0) if tracker.initialized else (0, 165, 255)
    cv2.putText(frame, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame
