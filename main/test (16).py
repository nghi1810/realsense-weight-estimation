import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d
import os
import time
import torch

TARGET_POINTS = 2048
BBOX_SIZE = 200

# =========================
# CREATE DIRS
# =========================
def farthest_point_sampling(points, k):
    N = points.shape[0]
    if N == 0:
        return np.array([])

    sampled_idx = np.zeros(k, dtype=int)
    distances = np.ones(N) * 1e10

    farthest = np.random.randint(0, N)

    for i in range(k):
        sampled_idx[i] = farthest
        centroid = points[farthest]

        dist = np.linalg.norm(points - centroid, axis=1)
        distances = np.minimum(distances, dist)

        farthest = np.argmax(distances)

    return points[sampled_idx]


def create_dirs(base_dir, current_id):
    folder_name = f"apple_{current_id}"
    save_root = os.path.join(base_dir, folder_name)

    # FULL
    rgb_raw_dir = os.path.join(save_root, "RGB_RAW")
    ply_full_dir = os.path.join(save_root, "PLY_FULL")
    depth_rgb_full_dir = os.path.join(save_root, "DEPTH_RGB_FULL")
    depth_npy_full = os.path.join(save_root, "DEPTH_FULL")

    # UNRAW
    rgb_unraw_dir = os.path.join(save_root, "RGB_UNRAW")
    ply_unraw_dir = os.path.join(save_root, "PLY_UNRAW")
    depth_rgb_unraw_dir = os.path.join(save_root, "DEPTH_RGB_UNRAW")
    depth_npy_unraw = os.path.join(save_root, "DEPTH_UNRAW")

    # BBOX
    rgb_bbox_dir = os.path.join(save_root, "RGB_BBOX")
    ply_bbox_dir = os.path.join(save_root, "PLY_BBOX")
    depth_rgb_bbox_dir = os.path.join(save_root, "DEPTH_RGB_BBOX")
    depth_npy_bbox = os.path.join(save_root, "DEPTH_BBOX")

    for d in [rgb_raw_dir, ply_full_dir, depth_rgb_full_dir, depth_npy_full,
              rgb_unraw_dir, ply_unraw_dir, depth_rgb_unraw_dir, depth_npy_unraw,
              rgb_bbox_dir, ply_bbox_dir, depth_rgb_bbox_dir, depth_npy_bbox]:
        os.makedirs(d, exist_ok=True)

    return (rgb_raw_dir, ply_full_dir, depth_rgb_full_dir, depth_npy_full,
            rgb_unraw_dir, ply_unraw_dir, depth_rgb_unraw_dir, depth_npy_unraw,
            rgb_bbox_dir, ply_bbox_dir, depth_rgb_bbox_dir, depth_npy_bbox,
            folder_name)


def load_json_config(device, json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)

    json_str = str(config).replace("'", '\"')
    device.as_rs400_advanced_mode().load_json(json_str)

# =========================
# MAIN
# =========================
def run_capture(model_path, base_dir="dataset"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.to(device)


    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    device = profile.get_device()
    print("Device:", device.get_info(rs.camera_info.name))

    # ===== ADVANCED MODE =====
    adv = rs.rs400_advanced_mode(device)

    if not adv.is_enabled():
        print("Enabling advanced mode...")
        pipeline.stop()

        adv.toggle_advanced_mode(True)
        time.sleep(2)

        # phải start lại pipeline sau khi reboot
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        device = profile.get_device()
        adv = rs.rs400_advanced_mode(device)

    # ===== LOAD JSON =====
    with open(r"C:\Users\phand\Desktop\new\setting12.json", 'r') as f:
        json_str = f.read()

    adv.load_json(json_str)
    print("JSON loaded!")

    align = rs.align(rs.stream.color)




    current_id = 1
    (rgb_raw_dir, ply_full_dir, depth_rgb_full_dir, depth_npy_full,
     rgb_unraw_dir, ply_unraw_dir, depth_rgb_unraw_dir, depth_npy_unraw,
     rgb_bbox_dir, ply_bbox_dir, depth_rgb_bbox_dir, depth_npy_bbox,
     folder_name) = create_dirs(base_dir, current_id)

    save_id = 0

    print("Press E = save | A = new folder | ESC = exit")

    try:
        while True:

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            results = model(color, conf=0.4, verbose=False)

            bbox = None

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])
                    name = model.names[cls]

                    if name == "apple":

                        # ===== FIX: bbox luôn 100x100 =====
                        x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0])

                        cx = (x1_yolo + x2_yolo) // 2
                        cy = (y1_yolo + y2_yolo) // 2

                        half = BBOX_SIZE // 2

                        x1 = max(0, cx - half)
                        y1 = max(0, cy - half)
                        x2 = min(color.shape[1], cx + half)
                        y2 = min(color.shape[0], cy + half)

                        bbox = (x1, y1, x2, y2)

                        cv2.rectangle(color,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(color,"apple_100x100",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            cv2.imshow("RGB", color)
            key = cv2.waitKey(1)

            if key == ord('a') or key == ord('A'):
                current_id += 1
                (rgb_raw_dir, ply_full_dir, depth_rgb_full_dir, depth_npy_full,
                 rgb_unraw_dir, ply_unraw_dir, depth_rgb_unraw_dir, depth_npy_unraw,
                 rgb_bbox_dir, ply_bbox_dir, depth_rgb_bbox_dir, depth_npy_bbox,
                 folder_name) = create_dirs(base_dir, current_id)
                print(f"Switched to {folder_name}")

            if key == ord('e'):

                last_npy_full, last_ply_full, last_rgb_full, last_depth_full = None, None, None, None
                last_npy_unraw, last_ply_2048, last_rgb_unraw, last_depth_unraw = None, None, None, None
                last_npy_bbox, last_ply_bbox, last_rgb_bbox, last_depth_bbox = None, None, None, None

                try:
                    timestamp = int(time.time())
                    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

                    # ================= FULL =================
                    points_full = []
                    h, w = color.shape[:2]

                    for y in range(h):
                        for x in range(w):
                            d = depth_frame.get_distance(x, y)
                            if d == 0:
                                continue
                            point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                            points_full.append(point)

                    if len(points_full) > 0:
                        points_full = np.array(points_full)

                        npy_path_full = os.path.join(depth_npy_full, f"{folder_name}_{save_id+1}.npy")
                        np.save(npy_path_full, points_full)

                        pcd_full = o3d.geometry.PointCloud()
                        pcd_full.points = o3d.utility.Vector3dVector(points_full)
                        ply_path_full = os.path.join(ply_full_dir, f"{folder_name}_{save_id+1}.ply")
                        o3d.io.write_point_cloud(ply_path_full, pcd_full)

                        cv2.imwrite(os.path.join(rgb_raw_dir, f"rgb_{timestamp}.png"), color)

                        depth_img = np.asanyarray(depth_frame.get_data())
                        depth_color_full = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_img, alpha=0.03),
                            cv2.COLORMAP_JET
                        )
                        cv2.imwrite(os.path.join(depth_rgb_full_dir, f"depth_{timestamp}.png"), depth_color_full)

                    # ================= UNRAW =================
                    # ================= UNRAW =================
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        points_unraw = []

                        for y in range(y1, y2):
                            for x in range(x1, x2):
                                d = depth_frame.get_distance(x, y)
                                if d == 0:
                                    continue
                                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                                points_unraw.append(point)

                        if len(points_unraw) > 0:
                            points_unraw = np.array(points_unraw)

                            # =========================
                            # FILTER PLANE & SAMPLE
                            # =========================
                            try:
                                center = np.mean(points_unraw, axis=0)
                                dist = np.linalg.norm(points_unraw - center, axis=1)
                                idx = np.argsort(dist)[-5000:]
                                far_points = points_unraw[idx]

                                z_sorted = far_points[np.argsort(far_points[:,2])]
                                plane_points = z_sorted[-4:]
                                p1, p2, p3, p4 = plane_points

                                normal = np.cross(p2 - p1, p3 - p1)
                                if np.linalg.norm(normal) == 0:
                                    raise Exception("Plane error")

                                d_plane = -np.dot(normal, p1)
                                dist_plane = np.abs(
                                    normal[0]*points_unraw[:,0] +
                                    normal[1]*points_unraw[:,1] +
                                    normal[2]*points_unraw[:,2] +
                                    d_plane
                                ) / np.linalg.norm(normal)

                                filtered_points = points_unraw[dist_plane > 0.03]
                                if len(filtered_points) == 0:
                                    raise Exception("No filtered points")

                                # =========================
                                # SAVE NPY_UNRAW
                                # =========================
                                npy_path = os.path.join(depth_npy_unraw, f"{folder_name}_{save_id+1}.npy")
                                np.save(npy_path, filtered_points)
                                print("Saved:", npy_path)

                                # =========================
                                # SAMPLE 2048 & SAVE PLY
                                # =========================
                                # SAMPLE 2048 bằng Open3D FPS (không duplicate)
                                n = len(filtered_points)

                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filtered_points)

                                if n >= TARGET_POINTS:
                                    pcd = pcd.farthest_point_down_sample(TARGET_POINTS)
                                    sampled = np.asarray(pcd.points)
                                else:
                                    # không đủ điểm → giữ nguyên, KHÔNG thêm
                                    sampled = filtered_points

                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(sampled)

                                ply_2048_path = os.path.join(ply_unraw_dir, f"{folder_name}_{save_id+1}_2048.ply")
                                o3d.io.write_point_cloud(ply_2048_path, pcd)
                                last_ply_2048 = ply_2048_path
                                print("Saved:", ply_2048_path)
                                # =========================
                                # SAVE RGB + DEPTH
                                # =========================
                                rgb_crop = color[y1:y2, x1:x2]
                                cv2.imwrite(os.path.join(rgb_unraw_dir, f"rgb_{timestamp}.png"), rgb_crop)

                                depth_img = np.asanyarray(depth_frame.get_data())
                                depth_crop = depth_img[y1:y2, x1:x2]
                                depth_color = cv2.applyColorMap(
                                    cv2.convertScaleAbs(depth_crop, alpha=0.03),
                                    cv2.COLORMAP_JET
                                )
                                cv2.imwrite(os.path.join(depth_rgb_unraw_dir, f"depth_{timestamp}.png"), depth_color)

                            except Exception as e:
                                print("UNRAW ERROR:", e)

                    # ================= BBOX =================
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        points_bbox = []

                        for y in range(y1, y2):
                            for x in range(x1, x2):
                                d = depth_frame.get_distance(x, y)
                                if d == 0:
                                    continue
                                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                                points_bbox.append(point)

                        if len(points_bbox) > 0:
                            points_bbox = np.array(points_bbox)

                            np.save(os.path.join(depth_npy_bbox, f"{folder_name}_{save_id+1}.npy"), points_bbox)

                            pcd_bbox = o3d.geometry.PointCloud()
                            pcd_bbox.points = o3d.utility.Vector3dVector(points_bbox)

                            o3d.io.write_point_cloud(
                                os.path.join(ply_bbox_dir, f"{folder_name}_{save_id+1}.ply"), pcd_bbox
                            )

                            cv2.imwrite(os.path.join(rgb_bbox_dir, f"rgb_{timestamp}.png"), color[y1:y2, x1:x2])

                            depth_img = np.asanyarray(depth_frame.get_data())
                            depth_crop = depth_img[y1:y2, x1:x2]
                            depth_color = cv2.applyColorMap(
                                cv2.convertScaleAbs(depth_crop, alpha=0.03),
                                cv2.COLORMAP_JET
                            )
                            cv2.imwrite(os.path.join(depth_rgb_bbox_dir, f"depth_{timestamp}.png"), depth_color)

                    save_id += 1

                except Exception as e:
                    print("ERROR:", e)

            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


MODEL_PATH = r"C:\Users\phand\Desktop\new\best.pt"
run_capture(MODEL_PATH, base_dir="data_test1")