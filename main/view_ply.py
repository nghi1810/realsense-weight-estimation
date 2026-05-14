import open3d as o3d
import numpy as np

# Load point cloud
ply_path = r"C:\Users\phand\Desktop\new\apple_5\PLY_in_BB_002.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)

# =========================
# TÍNH 5 ĐIỂM GẦN & XA NHẤT (so với gốc 0,0,0)
# =========================
points = np.asarray(pcd.points)

# Tính khoảng cách đến gốc
distances = np.linalg.norm(points, axis=1)

# Sắp xếp index theo khoảng cách
sorted_idx = np.argsort(distances)

# 5 điểm gần nhất
nearest_5 = points[sorted_idx[:500]]
print("\n5 điểm gần nhất:")
print(nearest_5)

# 5 điểm xa nhất
farthest_5 = points[sorted_idx[-500:]]
print("\n5 điểm xa nhất:")
print(farthest_5)

# =========================
# VISUALIZER (giữ nguyên)
# =========================
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
# 👉 THÊM DÒNG NÀY
vis.reset_view_point(True)

ctr = vis.get_view_control()

# 👉 Set hướng nhìn
ctr.set_front([0, 0, -1])
ctr.set_up([0, -1, 0])

# 👉 Zoom gần
ctr.set_zoom(0.1)

vis.run()
vis.destroy_window()