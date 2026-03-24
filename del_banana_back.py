import open3d as o3d
import numpy as np

TARGET_POINTS = 2048

pcd = o3d.io.read_point_cloud(r"C:\Users\phand\Desktop\New folder\apple_1.ply")
points = np.asarray(pcd.points)

print("Total:", len(points))

# tìm tâm
center = np.mean(points, axis=0)

# distance tới tâm
dist = np.linalg.norm(points - center, axis=1)

# lấy 200 point xa nhất
idx = np.argsort(dist)[-200:]
far_points = points[idx]

# chọn các point có z lớn nhất
z_sorted = far_points[np.argsort(far_points[:,2])]
plane_points = z_sorted[-4:]

p1, p2, p3, p4 = plane_points

# tính plane
v1 = p2 - p1
v2 = p3 - p1

normal = np.cross(v1, v2)
a, b, c = normal
d = -np.dot(normal, p1)

print("Plane:", a, b, c, d)

# distance tới plane
dist_plane = np.abs(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / np.linalg.norm(normal)

# bỏ mặt phẳng
mask = dist_plane > 0.01
filtered_points = points[mask]

print("After remove:", len(filtered_points))

# ==============================
# Đưa về đúng 2048 point
# ==============================

n = len(filtered_points)

if n > TARGET_POINTS:
    idx = np.random.choice(n, TARGET_POINTS, replace=False)
    sampled_points = filtered_points[idx]

elif n < TARGET_POINTS:
    idx = np.random.choice(n, TARGET_POINTS - n, replace=True)
    extra = filtered_points[idx]
    sampled_points = np.vstack((filtered_points, extra))

else:
    sampled_points = filtered_points

print("Final:", len(sampled_points))

pcd_new = o3d.geometry.PointCloud()
pcd_new.points = o3d.utility.Vector3dVector(sampled_points)

o3d.io.write_point_cloud(r"C:\Users\phand\Desktop\New folder\banana_1_2048.ply", pcd_new)
print("Saved: apple_1_2048.ply")

o3d.visualization.draw_geometries([pcd_new])