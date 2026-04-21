import os
import json
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "best_model_v2.pt")
LABEL_MAP_PATH = os.path.join(ROOT, "label_mapping_v2.json")
CLASSES_PATH = os.path.join(ROOT, "data", "MSASL_classes.json")
HAND_MODEL_PATH = os.path.join(ROOT, "models", "hand_landmarker.task")

VIDEO_PATH = os.path.join(ROOT, "demo_test_video1.mp4")

T = 64
INPUT_DIM = 252
HIDDEN = 256
NUM_LAYERS = 2
DROPOUT = 0.3

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode


# Sample frame indices
def sample_indices(n_frames: int, t: int):
  if n_frames <= 0:
    return np.zeros((t,), dtype=np.int32)
  if n_frames >= t:
    return np.linspace(0, n_frames - 1, t).round().astype(np.int32)
  base = np.arange(n_frames, dtype=np.int32)
  pad = np.full((t - n_frames,), n_frames - 1, dtype=np.int32)
  return np.concatenate([base, pad], axis=0)


# Normalize one hand
def normalize_one_hand(hand_feat):
  t = hand_feat.shape[0]
  hand = hand_feat.reshape(t, 21, 3).copy()

  for i in range(t):
    pts = hand[i]

    if np.allclose(pts, 0.0):
      continue

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
      scale = 1.0

    hand[i] = pts / scale

  return hand.reshape(t, 63).astype(np.float32)


# Build final feature
def build_feature(x):
  left = normalize_one_hand(x[:, :63])
  right = normalize_one_hand(x[:, 63:])
  x_norm = np.concatenate([left, right], axis=1)

  vel = np.zeros_like(x_norm, dtype=np.float32)
  vel[1:] = x_norm[1:] - x_norm[:-1]

  return np.concatenate([x_norm, vel], axis=1).astype(np.float32)


# Attention pooling
class AttentionPooling(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.score = nn.Linear(dim, 1)

  def forward(self, x):
    w = self.score(x)
    a = torch.softmax(w, dim=1)
    pooled = torch.sum(a * x, dim=1)
    return pooled


# Define model
class BiGRUAttention(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.input_proj = nn.Sequential(
      nn.LayerNorm(INPUT_DIM),
      nn.Linear(INPUT_DIM, INPUT_DIM),
      nn.ReLU(),
      nn.Dropout(DROPOUT),
    )
    self.gru = nn.GRU(
      input_size=INPUT_DIM,
      hidden_size=HIDDEN,
      num_layers=NUM_LAYERS,
      batch_first=True,
      bidirectional=True,
      dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
    )
    self.pool = AttentionPooling(HIDDEN * 2)
    self.head = nn.Sequential(
      nn.LayerNorm(HIDDEN * 2),
      nn.Dropout(DROPOUT),
      nn.Linear(HIDDEN * 2, HIDDEN),
      nn.ReLU(),
      nn.Dropout(DROPOUT),
      nn.Linear(HIDDEN, num_classes),
    )

  def forward(self, x):
    x = self.input_proj(x)
    out, _ = self.gru(x)
    pooled = self.pool(out)
    return self.head(pooled)


# Extract hand landmark sequence
def extract_video_feature(video_path, landmarker):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise RuntimeError("cannot_open_video")

  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
  picks = sample_indices(n_frames, T)
  frame_to_slots = {}

  for i, p in enumerate(picks.tolist()):
    frame_to_slots.setdefault(int(p), []).append(i)

  feat = np.zeros((T, 126), dtype=np.float32)
  max_pick = int(np.max(picks)) if len(picks) > 0 else -1
  fi = 0

  while fi <= max_pick:
    ok, frame_bgr = cap.read()
    if not ok:
      break

    if fi in frame_to_slots:
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
      result = landmarker.detect(mp_image)

      left = np.zeros((63,), dtype=np.float32)
      right = np.zeros((63,), dtype=np.float32)

      if result.hand_landmarks and result.handedness:
        for lm_list, handed_list in zip(result.hand_landmarks, result.handedness):
          handed = handed_list[0].category_name
          pts = []
          for lm in lm_list:
            pts.extend([lm.x, lm.y, lm.z])
          arr = np.array(pts, dtype=np.float32)

          if handed == "Left":
            left = arr
          elif handed == "Right":
            right = arr

      for slot in frame_to_slots[fi]:
        feat[slot, :63] = left
        feat[slot, 63:] = right

    fi += 1

  cap.release()
  return feat


# Load label mapping
def load_label_mapping():
  with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)
  idx_to_label = {int(k): int(v) for k, v in mapping["idx_to_label"].items()}
  return idx_to_label


# Load classes
def load_classes():
  with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    return json.load(f)


# Main inference
def main():
  idx_to_label = load_label_mapping()
  classes = load_classes()

  num_classes = len(idx_to_label)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BiGRUAttention(num_classes=num_classes).to(device)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
  model.eval()

  options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
  )

  with HandLandmarker.create_from_options(options) as landmarker:
    x = extract_video_feature(VIDEO_PATH, landmarker)

  x = build_feature(x)
  x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

  with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

  pred_idx = int(np.argmax(probs))
  pred_label = idx_to_label[pred_idx]
  pred_word = classes[pred_label]
  pred_conf = float(probs[pred_idx])

  topk = np.argsort(probs)[::-1][:5]

  print("video =", VIDEO_PATH)
  print("predicted_label =", pred_label)
  print("predicted_word =", pred_word)
  print("confidence =", pred_conf)
  print()
  print("top5 predictions:")
  for rank, idx in enumerate(topk, 1):
    label = idx_to_label[int(idx)]
    word = classes[label]
    conf = float(probs[int(idx)])
    print(f"{rank}. {word} (label={label}, prob={conf:.4f})")


if __name__ == "__main__":
  main()