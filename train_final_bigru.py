import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_CSV = os.path.join(ROOT, "features_filtered_train10_val1_test1.csv")

BATCH_SIZE = 96
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
INPUT_DIM = 252
HIDDEN = 256
NUM_LAYERS = 2
DROPOUT = 0.3
SEED = 42
PATIENCE = 8
LABEL_SMOOTHING = 0.05


# Set random seed
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


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


# Build feature
def build_feature(x):
  left = normalize_one_hand(x[:, :63])
  right = normalize_one_hand(x[:, 63:])
  x_norm = np.concatenate([left, right], axis=1)

  vel = np.zeros_like(x_norm, dtype=np.float32)
  vel[1:] = x_norm[1:] - x_norm[:-1]

  return np.concatenate([x_norm, vel], axis=1).astype(np.float32)


# Apply landmark augmentation
def augment_feature(x):
  x = x.copy()

  noise = np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)
  x = x + noise

  scale = np.random.uniform(0.95, 1.05)
  x = x * scale

  if np.random.rand() < 0.3:
    mask_len = np.random.randint(2, 6)
    start = np.random.randint(0, max(1, x.shape[0] - mask_len))
    x[start:start + mask_len] = 0.0

  if np.random.rand() < 0.3:
    shift = np.random.randint(-3, 4)
    x = np.roll(x, shift, axis=0)

  return x.astype(np.float32)


# Dataset
class ASLDataset(Dataset):
  def __init__(self, df, train_mode=False):
    self.df = df.reset_index(drop=True)
    self.train_mode = train_mode

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    x = np.load(row["feature_file"]).astype(np.float32)
    x = build_feature(x)

    if self.train_mode:
      x = augment_feature(x)

    y = int(row["label_mapped"])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


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


# BiGRU model
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


# Build sampler
def build_sampler(labels):
  class_counts = np.bincount(labels)
  weights = 1.0 / class_counts[labels]
  weights = torch.DoubleTensor(weights)
  sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
  return sampler


# Train one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
  model.train()
  total_loss = 0.0

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_loss += loss.item() * x.size(0)

  return total_loss / len(loader.dataset)


# Evaluate model
@torch.no_grad()
def evaluate(model, loader, device):
  model.eval()
  ys = []
  ps = []

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    pred = logits.argmax(dim=1)

    ys.extend(y.cpu().numpy().tolist())
    ps.extend(pred.cpu().numpy().tolist())

  acc = accuracy_score(ys, ps)
  return acc


# Main training
def main():
  set_seed(SEED)

  df = pd.read_csv(INDEX_CSV)

  print("original samples =", len(df))
  print()
  print("split counts:")
  print(df["split"].value_counts())
  print()

  valid_labels = sorted(df["label"].unique().tolist())
  label_to_idx = {label: i for i, label in enumerate(valid_labels)}
  idx_to_label = {i: label for label, i in label_to_idx.items()}
  df["label_mapped"] = df["label"].map(label_to_idx)

  train_df = df[df["split"] == "train"].copy()
  val_df = df[df["split"] == "val"].copy()
  test_df = df[df["split"] == "test"].copy()

  print("train samples =", len(train_df))
  print("val samples =", len(val_df))
  print("test samples =", len(test_df))
  print("num classes =", len(valid_labels))

  train_labels = train_df["label_mapped"].to_numpy()
  sampler = build_sampler(train_labels)

  train_loader = DataLoader(
    ASLDataset(train_df, train_mode=True),
    batch_size=BATCH_SIZE,
    sampler=sampler
  )
  val_loader = DataLoader(
    ASLDataset(val_df, train_mode=False),
    batch_size=BATCH_SIZE,
    shuffle=False
  )
  test_loader = DataLoader(
    ASLDataset(test_df, train_mode=False),
    batch_size=BATCH_SIZE,
    shuffle=False
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device =", device)

  model = BiGRUAttention(num_classes=len(valid_labels)).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
  )
  criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

  best_val_acc = 0.0
  best_model_path = os.path.join(ROOT, "best_model_v2.pt")
  label_map_path = os.path.join(ROOT, "label_mapping_v2.json")
  no_improve = 0

  for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_acc = evaluate(model, val_loader, device)
    scheduler.step(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:02d} | loss = {train_loss:.4f} | val_acc = {val_acc:.4f} | lr = {current_lr:.6f}")

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      no_improve = 0
      torch.save(model.state_dict(), best_model_path)

      with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
          {
            "label_to_idx": {str(k): v for k, v in label_to_idx.items()},
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()}
          },
          f,
          ensure_ascii=False,
          indent=2
        )
    else:
      no_improve += 1

    if no_improve >= PATIENCE:
      print("early stopping")
      break

  print("best_val_acc =", best_val_acc)

  model.load_state_dict(torch.load(best_model_path, map_location=device))
  test_acc = evaluate(model, test_loader, device)
  print("test_acc =", test_acc)
  print("saved model =", best_model_path)
  print("saved mapping =", label_map_path)


if __name__ == "__main__":
  main()