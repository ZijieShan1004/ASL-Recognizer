import json
import numpy as np
import torch
import torch.nn as nn

from shared_config import (
  MODEL_PATH,
  LABEL_MAP_PATH,
  CLASSES_PATH,
  INPUT_DIM,
  HIDDEN,
  NUM_LAYERS,
  DROPOUT,
)


class AttentionPooling(nn.Module):
  # Compute attention-weighted temporal pooling
  def __init__(self, dim):
    super().__init__()
    self.score = nn.Linear(dim, 1)

  # Pool sequence features into one vector
  def forward(self, x):
    w = self.score(x)
    a = torch.softmax(w, dim=1)
    pooled = torch.sum(a * x, dim=1)
    return pooled


class BiGRUAttention(nn.Module):
  # Build the BiGRU attention classifier
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

  # Run the classifier on one sequence batch
  def forward(self, x):
    x = self.input_proj(x)
    out, _ = self.gru(x)
    pooled = self.pool(out)
    return self.head(pooled)


# Normalize one hand trajectory
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


# Build normalized landmark plus velocity features
def build_feature(x):
  left = normalize_one_hand(x[:, :63])
  right = normalize_one_hand(x[:, 63:])
  x_norm = np.concatenate([left, right], axis=1)

  vel = np.zeros_like(x_norm, dtype=np.float32)
  vel[1:] = x_norm[1:] - x_norm[:-1]

  return np.concatenate([x_norm, vel], axis=1).astype(np.float32)


class ASLModelRuntime:
  # Initialize model, labels, and class names
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.idx_to_label = self._load_label_mapping()
    self.classes = self._load_classes()
    self.model = BiGRUAttention(num_classes=len(self.idx_to_label)).to(self.device)
    self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
    self.model.eval()

  # Load runtime label mapping
  def _load_label_mapping(self):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
      mapping = json.load(f)
    return {int(k): int(v) for k, v in mapping["idx_to_label"].items()}

  # Load MS-ASL class names
  def _load_classes(self):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
      return json.load(f)

  # Predict the top word for one raw landmark window
  def predict_window(self, raw_window):
    x = build_feature(raw_window)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    with torch.no_grad():
      logits = self.model(x)
      probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = self.idx_to_label[pred_idx]
    pred_word = self.classes[pred_label]
    pred_conf = float(probs[pred_idx])

    return pred_word, pred_conf, probs