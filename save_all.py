import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d
import os

TARGET_POINTS = 2048

folder_id = "apple_101"

model = YOLO("best.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

save_id = 0

os.makedirs(f"ply/{folder_id}", exist_ok=True)
os.makedirs(f"ply_2048/{folder_id}", exist_ok=True)
os.makedirs(f"npy/{folder_id}", exist_ok=True)

print("Press E to save apple pointcloud")

while True:

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    results = model(color, conf=0.5, verbose=False)

    bbox = None

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            name = model.names[cls]

            if name == "green_apple":

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                expand = 20

                x1 = max(0, x1 - expand)
                y1 = max(0, y1 - expand)
                x2 = min(color.shape[1], x2 + expand)
                y2 = min(color.shape[0], y2 + expand)

                bbox = (x1, y1, x2, y2)

                cv2.rectangle(color,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(color,"apple",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("RGB", color)

    key = cv2.waitKey(1)

    if key == ord('e') and bbox is not None:

        x1,y1,x2,y2 = bbox

        points = []

        for y in range(y1,y2):
            for x in range(x1,x2):

                d = depth_frame.get_distance(x,y)

                if d == 0:
                    continue

                point = rs.rs2_deproject_pixel_to_point(
                    depth_frame.profile.as_video_stream_profile().intrinsics,
                    [x,y],
                    d
                )

                r,g,b = color[y,x]

                points.append([point[0],point[1],point[2],r,g,b])

        filename = f"ply/{folder_id}/apple_{save_id+1}.ply"

        with open(filename,'w') as f:

            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")

        print("Saved:", filename)

        pcd = o3d.io.read_point_cloud(filename)
        points_np = np.asarray(pcd.points)

        center = np.mean(points_np, axis=0)
        dist = np.linalg.norm(points_np - center, axis=1)

        idx = np.argsort(dist)[-1000:]
        far_points = points_np[idx]

        z_sorted = far_points[np.argsort(far_points[:,2])]
        plane_points = z_sorted[-4:]

        p1, p2, p3, p4 = plane_points

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) == 0:
            print("Plane error")
            os.remove(filename)
            print("Deleted:", filename)
            continue

        a, b, c = normal
        d = -np.dot(normal, p1)

        dist_plane = np.abs(
            a*points_np[:,0] + b*points_np[:,1] + c*points_np[:,2] + d
        ) / np.linalg.norm(normal)

        mask = dist_plane > 0.02
        filtered_points = points_np[mask]

        print("After remove:", len(filtered_points))

        n = len(filtered_points)

        if n == 0:

            print("ERROR: no points left")

            os.remove(filename)

            print("Deleted:", filename)

            continue

        npy_name = f"npy/{folder_id}/apple_{save_id+1}.npy"
        np.save(npy_name, filtered_points)

        print("Saved:", npy_name)

        if n > TARGET_POINTS:
            idx = np.random.choice(n, TARGET_POINTS, replace=False)
            sampled_points = filtered_points[idx]

        elif n < TARGET_POINTS:
            idx = np.random.choice(n, TARGET_POINTS - n, replace=True)
            extra = filtered_points[idx]
            sampled_points = np.vstack((filtered_points, extra))

        else:
            sampled_points = filtered_points

        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(sampled_points)

        out_name = f"ply_2048/{folder_id}/apple_{save_id+1}_2048.ply"

        o3d.io.write_point_cloud(out_name, pcd_new)

        print("Saved:", out_name)

        save_id += 1

    if key == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()