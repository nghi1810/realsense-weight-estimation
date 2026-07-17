import numpy as np
import open3d as o3d

data = np.load(r"C:\Users\phand\Desktop\new\pointcloud_capture\npy_full\full_20260405_161305.npy")

print("Shape:", data.shape)
print("First 5 points:")
print(data[:5])

points = data[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])