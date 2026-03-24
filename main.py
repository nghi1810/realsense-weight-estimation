import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import tensorflow as tf
import joblib
from ultralytics import YOLO

# load models
yolo = YOLO("best.pt")
model = tf.keras.models.load_model(r"C:\Users\phand\Desktop\New folder2\apple_weight.weights.h5")
scaler = joblib.load("scaler.save")

def normalize_point_cloud(points):

    centroid = np.mean(points, axis=0)
    points = points - centroid

    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_dist

    return points


# RealSense
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

try:

    while True:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        results = yolo(frame, imgsz=640)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                points = []

                for v in range(y1, y2):
                    for u in range(x1, x2):

                        depth = depth_frame.get_distance(u, v)

                        if depth == 0:
                            continue

                        point = rs.rs2_deproject_pixel_to_point(
                            intr, [u, v], depth
                        )

                        points.append(point)

                if len(points) < 300:
                    continue

                points_np = np.array(points)

                # ---------- plane remove (giữ nguyên code bạn) ----------

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
                    continue

                a, b, c = normal
                d = -np.dot(normal, p1)

                dist_plane = np.abs(
                    a*points_np[:,0] + b*points_np[:,1] + c*points_np[:,2] + d
                ) / np.linalg.norm(normal)

                mask = dist_plane > 0.02
                filtered_points = points_np[mask]

                if len(filtered_points) < 50:
                    continue

                # ---------- sample 2048 ----------

                if len(filtered_points) >= 2048:
                    idx = np.random.choice(len(filtered_points), 2048, replace=False)
                else:
                    idx = np.random.choice(len(filtered_points), 2048, replace=True)

                sampled_points = filtered_points[idx]

                # ---------- normalize ----------

                sampled_points = normalize_point_cloud(sampled_points)

                sampled_points = np.expand_dims(sampled_points, axis=0)

                # ---------- predict ----------

                y_pred = model.predict(sampled_points, verbose=0)

                weight = scaler.inverse_transform(y_pred).flatten()[0]

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,
                            f"{weight:.2f} g",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2)

        cv2.imshow("Apple Weight Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()