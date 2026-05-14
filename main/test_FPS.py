import numpy as np
import open3d as o3d


def farthest_point_sampling(points, num_samples, deterministic=False):
    N, _ = points.shape

    if N == 0:
        return np.zeros((num_samples, 3), dtype=np.float32)

    sampled_idx = np.zeros(num_samples, dtype=int)
    distances = np.ones(N, dtype=np.float32) * 1e10

    if deterministic:
        # chọn điểm gần centroid nhất
        centroid = np.mean(points, axis=0)
        dist_to_center = np.sum((points - centroid) ** 2, axis=1)
        farthest = np.argmin(dist_to_center)
    else:
        farthest = np.random.randint(0, N)

    sampled_idx[0] = farthest

    for i in range(1, num_samples):
        diff = points - points[farthest]
        dist = np.sum(diff * diff, axis=1)

        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
        sampled_idx[i] = farthest

    return points[sampled_idx]


# ====== LOAD PLY ======
ply_path = r"C:\Users\phand\Desktop\new\pointcloud_capture\apple_1\PLY_in_contour\PLY_in_contour_001.ply"

pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)

print("Before FPS:", points.shape)

# ====== APPLY FPS ======
num_samples = 256  # bạn chỉnh tùy ý (256 / 512 / 1024)
sampled_points = farthest_point_sampling(points, num_samples, deterministic=True)

print("After FPS:", sampled_points.shape)

# ====== VISUALIZE ======

# Before (màu xám)
pcd_before = o3d.geometry.PointCloud()
pcd_before.points = o3d.utility.Vector3dVector(points)
pcd_before.paint_uniform_color([0.7, 0.7, 0.7])

# After (màu đỏ)
pcd_after = o3d.geometry.PointCloud()
pcd_after.points = o3d.utility.Vector3dVector(sampled_points)
pcd_after.paint_uniform_color([1, 0, 0])

# Dịch after sang bên phải để nhìn rõ
pcd_after.translate((0.1, 0, 0))

# Show
o3d.visualization.draw_geometries([pcd_before, pcd_after])