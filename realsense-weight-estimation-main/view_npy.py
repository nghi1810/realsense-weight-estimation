import numpy as np
import open3d as o3d

data = np.load(r"C:\Users\phand\Desktop\New folder\npy\apple_43\apple_19.npy")

print("Shape:", data.shape)
print("First 5 points:")
print(data[:5])

points = data[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])