import os
import torch
import open3d as o3d
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def plot_ply_one_by_one(folder, max_files=None, max_points=5000):
    files = [f for f in os.listdir(folder) if f.endswith(".ply")]
    files.sort()

    if max_files:
        files = files[:max_files]

    for fname in files:
        path = os.path.join(folder, fname)

        print(f"\nViewing: {fname}")

        pcd = o3d.io.read_point_cloud(path)

        pts = torch.tensor(
            np.asarray(pcd.points),   # ⚠ fix warning luôn
            dtype=torch.float32
        ).to(device)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        if pts.shape[0] > 0:

            # ===== downsample =====
            if pts.shape[0] > max_points:
                idx = torch.randperm(pts.shape[0], device=device)[:max_points]
                pts = pts[idx]

            pts_cpu = pts.cpu()

            ax.scatter(
                pts_cpu[:,0],
                pts_cpu[:,1],
                pts_cpu[:,2],
                s=1
            )

        else:
            ax.set_title(f"{fname} - EMPTY")

        ax.set_title(fname)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        plt.show()

        input("Press Enter to continue...")

# ===== RUN =====
import numpy as np

plot_ply_one_by_one("apple_5", max_files=50)
plot_ply_one_by_one("apple_6", max_files=50)
