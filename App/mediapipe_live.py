import cv2
import numpy as np
import mediapipe as mp

from shared_config import (
  HAND_MODEL_PATH,
  NUM_HANDS,
  MIN_HAND_DETECTION_CONFIDENCE,
  MIN_HAND_PRESENCE_CONFIDENCE,
  MIN_TRACKING_CONFIDENCE,
)


HAND_CONNECTIONS = [
  (0, 1), (1, 2), (2, 3), (3, 4),
  (0, 5), (5, 6), (6, 7), (7, 8),
  (5, 9), (9, 10), (10, 11), (11, 12),
  (9, 13), (13, 14), (14, 15), (15, 16),
  (13, 17), (17, 18), (18, 19), (19, 20),
  (0, 17)
]


class LiveHandExtractor:
  # Initialize MediaPipe hand landmarker
  def __init__(self):
    self.BaseOptions = mp.tasks.BaseOptions
    self.vision = mp.tasks.vision
    self.HandLandmarker = self.vision.HandLandmarker
    self.HandLandmarkerOptions = self.vision.HandLandmarkerOptions
    self.RunningMode = self.vision.RunningMode

    options = self.HandLandmarkerOptions(
      base_options=self.BaseOptions(model_asset_path=HAND_MODEL_PATH),
      running_mode=self.RunningMode.IMAGE,
      num_hands=NUM_HANDS,
      min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
      min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
      min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    self.landmarker = self.HandLandmarker.create_from_options(options)
    self.connections = HAND_CONNECTIONS

  # Convert landmark list to flat xyz array
  def _landmarks_to_array(self, lm_list):
    pts = []
    for lm in lm_list:
      pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts, dtype=np.float32)

  # Draw one detected hand on frame
  def _draw_hand(self, frame_bgr, lm_list):
    h, w = frame_bgr.shape[:2]
    points = []

    for lm in lm_list:
      x = int(round(lm.x * w))
      y = int(round(lm.y * h))
      points.append((x, y))
      cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

    for a, b in self.connections:
      if a < len(points) and b < len(points):
        cv2.line(frame_bgr, points[a], points[b], (255, 0, 0), 2)

  # Extract raw 126-dim hand feature from one frame
  def process_frame(self, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = self.landmarker.detect(mp_image)

    left = np.zeros((63,), dtype=np.float32)
    right = np.zeros((63,), dtype=np.float32)
    annotated = frame_bgr.copy()
    has_hands = False

    if result.hand_landmarks and result.handedness:
      has_hands = True
      for lm_list, handed_list in zip(result.hand_landmarks, result.handedness):
        handed = handed_list[0].category_name
        arr = self._landmarks_to_array(lm_list)

        if handed == "Left":
          left = arr
        elif handed == "Right":
          right = arr

        self._draw_hand(annotated, lm_list)

    feat = np.concatenate([left, right], axis=0).astype(np.float32)
    return feat, has_hands, annotated

  # Release MediaPipe resources
  def close(self):
    self.landmarker.close()