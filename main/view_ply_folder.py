import os
import open3d as o3d
import numpy as np

folder = r"C:\Users\phand\Desktop\new\apple_27"  # sửa lại path

# lấy danh sách file .ply
files = [f for f in os.listdir(folder) if f.endswith(".ply")]
files.sort()

print(f"Total files: {len(files)}")

for i, file in enumerate(files):
    path = os.path.join(folder, file)
    print(f"\n[{i+1}/{len(files)}] Viewing: {file}")

    # load point cloud
    pcd = o3d.io.read_point_cloud(path)

    # check file lỗi
    num_points = len(pcd.points)
    print("Number of points:", num_points)

    if num_points == 0:
        print("⚠️ File rỗng hoặc lỗi, bỏ qua")
        continue

    # =========================
    # FIX HƯỚNG (flip trục Z)
    # =========================
    pcd.transform([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # =========================
    # VISUALIZE TỪNG FILE
    # =========================
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=file, width=800, height=600)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()  # quan trọng để tránh bug file sau