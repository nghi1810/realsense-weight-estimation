import open3d as o3d

# Đường dẫn file ply
ply_path = r"C:\Users\phand\Desktop\New folder2\ply_2048\apple_251\apple_1_2048.ply"

# Load point cloud
pcd = o3d.io.read_point_cloud(ply_path)

# In thông tin
print(pcd)

# Hiển thị
o3d.visualization.draw_geometries([pcd])