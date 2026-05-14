# test_model.py — standalone inference script
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# PREPROCESSING (must match training EXACTLY)
# ============================================================
NUM_POINTS = 8192
FPS_SEED   = 42

def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    return points

def farthest_point_sampling(points, num_points=8192, seed=42):
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")

    N = points.shape[0]
    rng = np.random.default_rng(seed)

    if N < num_points:
        raise ValueError(f"Point cloud has only {N} points, cannot sample {num_points}")

    sampled_idx = np.zeros(num_points, dtype=np.int32)
    distances = np.full(N, np.inf)

    farthest = rng.integers(0, N)

    for i in range(num_points):
        sampled_idx[i] = farthest
        centroid = points[farthest]

        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)

        farthest = np.argmax(distances)

    return points[sampled_idx]

# ============================================================
# MODEL DEFINITION (must match training EXACTLY)
# ============================================================
def matmul(a, b):
    return torch.bmm(a, b)

def orthogonality_loss(transform, k, lam=0.001):
    B = transform.shape[0]
    eye = torch.eye(k, device=transform.device).unsqueeze(0).expand(B, -1, -1)
    aat = torch.bmm(transform, transform.transpose(1, 2))
    sq_frob = torch.sum((eye - aat) ** 2, dim=(1, 2))
    return lam * sq_frob.mean()

class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 32, 1)
        self.bn1   = nn.BatchNorm1d(32, momentum=0.1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.bn2   = nn.BatchNorm1d(64, momentum=0.1)
        self.conv3 = nn.Conv1d(64, 512, 1)
        self.bn3   = nn.BatchNorm1d(512, momentum=0.1)

        self.fc1 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.1)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128, momentum=0.1)

        self.fc3 = nn.Linear(128, k * k)
        nn.init.zeros_(self.fc3.weight)
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).flatten())

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1, self.k, self.k)

class PointNet(nn.Module):
    def __init__(self, num_points=8192, dropout=0.15, orth_lam=0.001):
        super().__init__()
        self.num_points = num_points
        self.orth_lam   = orth_lam

        self.tnet1 = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 32, 1);  self.bn1 = nn.BatchNorm1d(32, momentum=0.1)
        self.conv2 = nn.Conv1d(32, 32, 1); self.bn2 = nn.BatchNorm1d(32, momentum=0.1)

        self.tnet2 = TNet(k=32)

        self.conv3 = nn.Conv1d(32, 32, 1);  self.bn3 = nn.BatchNorm1d(32,  momentum=0.1)
        self.conv4 = nn.Conv1d(32, 64, 1);  self.bn4 = nn.BatchNorm1d(64,  momentum=0.1)
        self.conv5 = nn.Conv1d(64, 512, 1); self.bn5 = nn.BatchNorm1d(512, momentum=0.1)

        self.fc1     = nn.Linear(512, 256); self.fc_bn1 = nn.BatchNorm1d(256, momentum=0.1)
        self.drop1   = nn.Dropout(dropout)
        self.fc2     = nn.Linear(256, 128); self.fc_bn2 = nn.BatchNorm1d(128, momentum=0.1)
        self.drop2   = nn.Dropout(dropout)
        self.fc3     = nn.Linear(128, 1)

        self.aux_loss = 0.0

    def forward(self, x):
        # x: (B, N, 3)
        x_chans = x.transpose(1, 2)
        T1 = self.tnet1(x_chans)
        x  = matmul(x, T1)

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        T2 = self.tnet2(x)
        if self.training:
            self.aux_loss = orthogonality_loss(T2, k=32, lam=self.orth_lam)
        else:
            self.aux_loss = 0.0
        x = matmul(x.transpose(1, 2), T2).transpose(1, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x, _ = torch.max(x, dim=2)

        x = self.drop1(F.relu(self.fc_bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.fc_bn2(self.fc2(x))))

        return self.fc3(x)

# ============================================================
# PATHS — edit these
# ============================================================
MODEL_PATH  = r"C:\Users\phand\Desktop\new\best_model.pt"
SCALER_PATH = r"C:\Users\phand\Desktop\new\scaler.pkl"
PLY_PATH    = r"C:\Users\phand\Desktop\new\PLY_in_BB_023 (1).ply"

# ============================================================
# 1. LOAD MODEL
# ============================================================
t_total_start = time.perf_counter()

t0 = time.perf_counter()
model = PointNet(num_points=NUM_POINTS, dropout=0.15, orth_lam=0.001).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)
# best_model.pt was saved as a dict with extra fields; final_model.pt is just a state_dict
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?')})")
else:
    model.load_state_dict(ckpt)
    print("Loaded raw state_dict")

model.eval()
t_load_model = time.perf_counter() - t0

# ============================================================
# 2. LOAD SCALER
# ============================================================
t0 = time.perf_counter()
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
t_load_scaler = time.perf_counter() - t0

# ============================================================
# 3. PREPROCESS POINT CLOUD (must match training)
# ============================================================
t0 = time.perf_counter()
pcd    = o3d.io.read_point_cloud(PLY_PATH)
points = np.asarray(pcd.points)
print(f"Raw points: {points.shape}")

points = normalize_point_cloud(points)
points = farthest_point_sampling(points, NUM_POINTS, seed=FPS_SEED)
points = points.astype(np.float32)[None, ...]   # (1, 8192, 3)
t_preprocess = time.perf_counter() - t0

# ============================================================
# 4. PREDICT (standardized space)
# ============================================================
# warm-up GPU/cuDNN nếu xài CUDA — first forward thường chậm do init
if device.type == "cuda":
    with torch.no_grad():
        _ = model(torch.from_numpy(points).to(device))
    torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    pred_std = model(torch.from_numpy(points).to(device)).cpu().numpy()
if device.type == "cuda":
    torch.cuda.synchronize()
t_predict = time.perf_counter() - t0

# ============================================================
# 5. INVERSE TRANSFORM → REAL WEIGHT
# ============================================================
t0 = time.perf_counter()
pred_real = scaler.inverse_transform(pred_std.reshape(-1, 1)).flatten()
t_inverse = time.perf_counter() - t0

t_total = time.perf_counter() - t_total_start

print(f"Predicted weight: {pred_real[0]:.2f}")

# ============================================================
# TIMING REPORT
# ============================================================
print("\n===== TIMING (ms) =====")
print(f"Load model       : {t_load_model*1000:8.2f} ms")
print(f"Load scaler      : {t_load_scaler*1000:8.2f} ms")
print(f"Preprocess (FPS) : {t_preprocess*1000:8.2f} ms   <-- thường là phần chậm nhất")
print(f"Predict (forward): {t_predict*1000:8.2f} ms")
print(f"Inverse transform: {t_inverse*1000:8.2f} ms")
print(f"-----------------------------")
print(f"TOTAL            : {t_total*1000:8.2f} ms  ({t_total:.3f} s)")