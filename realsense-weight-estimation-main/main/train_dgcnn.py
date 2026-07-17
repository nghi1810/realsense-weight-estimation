# -*- coding: utf-8 -*-
"""
DGCNN_MAIN_FIXED.py
Fixes applied:
  1. Best model checkpoint (save/load best weights)
  2. Metrics computed from best epoch, not last
  3. Data augmentation for training (rotation + jitter + scale)
  4. Offline FPS caching (precompute & reuse sampled points)
  5. drop_last=True on val_loader to avoid single-sample BatchNorm crash
  6. Colab drive mount guarded with try/except
  7. Duplicate NUM_POINTS removed
  8. MAX_FILES_PER_FOLDER as top-level constant
"""

# ============================================================
# 0. GOOGLE DRIVE MOUNT (safe outside Colab)
# ============================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ============================================================
# 1. HYPERPARAMETERS (all in one place)
# ============================================================
EPOCHS               = 200
NUM_POINTS           = 1024
BATCH_SIZE           = 32
VAL_RATIO            = 0.2
MAX_FILES_PER_FOLDER = 50          # FIX 8: was hardcoded in class
K_NEIGHBORS          = 20
CHECKPOINT_PATH      = "best_model.pth"

DATASET_PATH = "/content/drive/MyDrive/pointcloud_capture"

# ============================================================
# 2. IMPORTS
# ============================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# 3. LABELS
# ============================================================
labels_raw = [
    49.84,62.31,49.22,54.05,57.98,58.87,54.37,42.15,52.67,57.97,
    52.24,51.98,50.35,50.1,49.95,51.94,48.46,50.16,50.27,43.17,
    58.43,48.8,48.23,48.83,47.74,45.96,63.28,56.8,48.35,51.57,
    50.8,56.12,49.03,50.24,49.66,49.19,48.15,47.94,50.81,57.74,
    42.63,44.92,45.99,49.89,50.29,48.74,45.92,44.65,46.61,49.87,
    47.42,46.14,46.48,51.71,49.12,54.89,49.91,57.74,61.51,44.97,
    50.92,47.14,58.99,47.5,48.29,53.57,59.93,59.6,51.89,56.23,
    46.81,46.2,47.71,46.51,57.28,52.79,50.11,49.44,51.13,49.85,
    41.07,52.17,55.91,48.16,47.18,58.74,49.51,54.99,52.7,48.82,
    52.53,51.29,57.24,44.06,63.14,53.14,48.53,62.54,49.41,55.76,
    47.54,53.84,46.5,48.58,45.04,46.48,58.52,43.43,45.45,50.01,
    49.42,43.87,44.9,46.65,49.2,45.28,49.61,45.82,52.41,50.19,
    54.03,46.23,45.74,52.5,46.35,46.41,47.04,46.46
]
labels_raw = np.array(labels_raw)
print(f"Total labels: {len(labels_raw)}")

# ============================================================
# 4. RANDOM SEED
# ============================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 5. FOLDER → LABEL MAPPING & TRAIN/VAL SPLIT
# ============================================================
folders = sorted(
    [f for f in os.listdir(DATASET_PATH) if f.startswith("apple_")],
    key=lambda x: int(x.split("_")[1])
)
assert len(folders) == len(labels_raw), \
    f"Mismatch: {len(folders)} folders vs {len(labels_raw)} labels"

folder_to_label_raw = {f: l for f, l in zip(folders, labels_raw)}

all_folders = list(folder_to_label_raw.keys())
random.shuffle(all_folders)

num_val      = max(1, int(len(all_folders) * VAL_RATIO))
val_folders  = all_folders[:num_val]
train_folders = all_folders[num_val:]

print(f"Train: {len(train_folders)} folders | Val: {len(val_folders)} folders")

# FIX 1 – normalize using TRAIN stats only
train_labels = np.array([folder_to_label_raw[f] for f in train_folders])
mean = train_labels.mean()
std  = train_labels.std()
print(f"Train mean: {mean:.4f} | Train std: {std:.4f}")

folder_to_label = {
    f: (folder_to_label_raw[f] - mean) / std
    for f in (train_folders + val_folders)
}

# ============================================================
# 6. FARTHEST POINT SAMPLING (with disk cache)
# ============================================================
def farthest_point_sampling(points: np.ndarray, num_samples: int,
                             deterministic: bool = False) -> np.ndarray:
    N = points.shape[0]
    if N == 0:
        return np.zeros((num_samples, 3), dtype=np.float32)

    sampled_idx = np.zeros(num_samples, dtype=int)
    distances   = np.full(N, 1e10, dtype=np.float32)
    farthest    = 0 if deterministic else np.random.randint(0, N)
    sampled_idx[0] = farthest

    for i in range(1, num_samples):
        diff       = points - points[farthest]
        dist       = (diff * diff).sum(axis=1)
        distances  = np.minimum(distances, dist)
        farthest   = int(np.argmax(distances))
        sampled_idx[i] = farthest

    return points[sampled_idx]


def load_points_cached(ply_path: str, num_points: int,
                       deterministic: bool) -> np.ndarray:
    """
    FIX 4 – Cache FPS result as .npy next to the .ply file.
    On first call: read PLY → FPS → save cache.
    On later calls: load cache directly (much faster).
    """
    suffix   = f"_fps{num_points}_{'det' if deterministic else 'rand'}.npy"
    npy_path = ply_path.replace(".ply", suffix)

    if os.path.exists(npy_path):
        return np.load(npy_path)

    pcd    = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    if points.shape[0] == 0:
        print(f"[WARNING] Empty point cloud: {ply_path}")
        points = np.zeros((num_points, 3), dtype=np.float32)
    elif points.shape[0] < num_points:
        choice = np.random.choice(points.shape[0], num_points, replace=True)
        points = points[choice]
    else:
        points = farthest_point_sampling(points, num_points, deterministic)

    np.save(npy_path, points)
    return points

# ============================================================
# 7. DATASET
# ============================================================
class PointCloudDataset(Dataset):
    def __init__(self, folder_list, folder_to_label, base_path,
                 num_points=1024, split='train',
                 max_files_per_folder=MAX_FILES_PER_FOLDER):

        self.num_points = num_points
        self.split      = split
        self.samples    = []

        for folder in folder_list:
            folder_path = os.path.join(base_path, folder, "PLY_in_contour")
            if not os.path.isdir(folder_path):
                continue

            ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
            if len(ply_files) > max_files_per_folder:
                ply_files = (random.sample(ply_files, max_files_per_folder)
                             if split == 'train'
                             else sorted(ply_files)[:max_files_per_folder])

            for ply_file in ply_files:
                self.samples.append((
                    os.path.join(folder_path, ply_file),
                    folder_to_label[folder]
                ))

    def __len__(self):
        return len(self.samples)

    # FIX 3 – data augmentation (train only)
    @staticmethod
    def augment(points: np.ndarray) -> np.ndarray:
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(theta), np.sin(theta)
        R     = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]], dtype=np.float32)
        points = points @ R.T

        # Random scale
        scale  = np.random.uniform(0.9, 1.1)
        points = points * scale

        # Gaussian jitter
        points = points + np.random.randn(*points.shape).astype(np.float32) * 0.01

        return points

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        deterministic    = (self.split != 'train')

        points = load_points_cached(file_path, self.num_points, deterministic)

        # Centre
        points = points - points.mean(axis=0)

        if self.split == 'train':
            points = self.augment(points)

        return (torch.from_numpy(points).float(),
                torch.tensor(label, dtype=torch.float32))


# ============================================================
# 8. DATALOADERS
# ============================================================
train_dataset = PointCloudDataset(train_folders, folder_to_label,
                                  DATASET_PATH, NUM_POINTS, 'train')
val_dataset   = PointCloudDataset(val_folders,   folder_to_label,
                                  DATASET_PATH, NUM_POINTS, 'val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2,
                          drop_last=True)   # FIX 5: prevents 1-sample batch

# ============================================================
# 9. DGCNN MODEL
# ============================================================
def knn(x, k):
    inner            = -2 * torch.matmul(x.transpose(2, 1), x)
    xx               = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_dist    = -xx - inner - xx.transpose(2, 1)
    return pairwise_dist.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20):
    B, C, N  = x.size()
    idx      = knn(x, k=k)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx      = (idx + idx_base).view(-1)

    x_t     = x.transpose(2, 1).contiguous()
    feature = x_t.view(B * N, -1)[idx].view(B, N, k, C)
    x_t     = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)
    return torch.cat((feature - x_t, x_t), dim=3).permute(0, 3, 1, 2)


class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, k=K_NEIGHBORS):
        x = get_graph_feature(x, k)
        x = self.mlp(x)
        return x.max(dim=-1)[0]


class DGCNN_Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EdgeConv(3,   64)
        self.conv2 = EdgeConv(64,  64)
        self.conv3 = EdgeConv(64,  128)
        self.conv4 = EdgeConv(128, 256)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x  = x.permute(0, 2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x  = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        x  = torch.cat([F.adaptive_max_pool1d(x, 1).squeeze(-1),
                         F.adaptive_avg_pool1d(x, 1).squeeze(-1)], dim=1)
        return self.fc(x)


# ============================================================
# 10. TRAINING SETUP
# ============================================================
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = DGCNN_Regressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)


class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-3):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


early_stopping = EarlyStopping(patience=50, min_delta=1e-3)


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for points, labels_batch in loader:
        points, labels_batch = points.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        preds = model(points)
        loss  = criterion(preds.view(-1), labels_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for points, labels_batch in loader:
            points, labels_batch = points.to(device), labels_batch.to(device)
            preds     = model(points)
            loss      = criterion(preds.view(-1), labels_batch.view(-1))
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_batch.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_loss / len(loader), all_preds, all_labels


# ============================================================
# 11. TRAINING LOOP
# ============================================================
train_losses, val_losses = [], []
val_maes, val_mses, val_rmses, val_r2s = [], [], [], []

best_val_loss = float('inf')                          # FIX 1 – checkpoint tracking
best_preds    = best_labels = None                    # FIX 2 – best-epoch metrics

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, all_preds, all_labels = evaluate(model, val_loader)

    # Denormalize
    all_preds_orig  = all_preds  * std + mean
    all_labels_orig = all_labels * std + mean

    mse  = mean_squared_error(all_labels_orig, all_preds_orig)
    mae  = mean_absolute_error(all_labels_orig, all_preds_orig)
    rmse = np.sqrt(mse)
    r2   = r2_score(all_labels_orig, all_preds_orig)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_maes.append(mae)
    val_mses.append(mse)
    val_rmses.append(rmse)
    val_r2s.append(r2)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:03d} | Train={train_loss:.4f} | Val={val_loss:.4f} "
          f"| MAE={mae:.4f} | R2={r2:.4f} | LR={current_lr:.6f}")

    # FIX 1 – save best model
    if val_loss < best_val_loss:
        best_val_loss   = val_loss
        best_preds      = all_preds_orig.copy()   # FIX 2
        best_labels_out = all_labels_orig.copy()  # FIX 2
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f"  LR changed: {current_lr:.6f} → {new_lr:.6f}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# ============================================================
# 12. LOAD BEST MODEL BEFORE FINAL EVALUATION
# ============================================================
print(f"\nLoading best checkpoint from '{CHECKPOINT_PATH}' ...")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# FIX 2 – use best-epoch predictions for final scatter plot
y_true = best_labels_out.flatten()
y_pred = best_preds.flatten()
mae    = mean_absolute_error(y_true, y_pred)
rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
r2     = r2_score(y_true, y_pred)
print(f"\nBest epoch → MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

# ============================================================
# 13. PLOTS
# ============================================================
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses,   label="Val Loss")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid()

plt.subplot(2, 3, 2)
plt.plot(epochs_range, val_maes, label="MAE")
plt.title("MAE"); plt.xlabel("Epoch"); plt.ylabel("MAE")
plt.legend(); plt.grid()

plt.subplot(2, 3, 3)
plt.plot(epochs_range, val_r2s, label="R2")
plt.title("R2 Score"); plt.xlabel("Epoch"); plt.ylabel("R2")
plt.legend(); plt.grid()

plt.subplot(2, 3, 4)
plt.plot(epochs_range, val_rmses, label="RMSE")
plt.title("RMSE"); plt.xlabel("Epoch"); plt.ylabel("RMSE")
plt.legend(); plt.grid()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# Scatter plot (uses best-epoch preds, not last-epoch)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
x_line  = np.linspace(min_val, max_val, 100)

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot(x_line, x_line,       'r--', label='Ideal (y=x)')
plt.plot(x_line, x_line + mae, 'g--', label=f'+MAE ({mae:.2f})')
plt.plot(x_line, x_line - mae, 'g--', label=f'-MAE ({mae:.2f})')
plt.xlabel("Ground Truth"); plt.ylabel("Prediction")
plt.title(f"Prediction vs Ground Truth (Best Epoch)\nR2={r2:.3f}, RMSE={rmse:.3f}")
plt.legend(); plt.grid(True)
plt.savefig("scatter_best_epoch.png", dpi=150)
plt.show()