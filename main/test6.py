import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pyrealsense2 as rs
import os
import datetime
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Device: {device}")

model = YOLO("apple.pt")
model.to(device)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# ===== LOAD CONFIG FILE (JSON PRESET) =====
preset_file = r"C:\Users\phand\Desktop\new\setting14.json"

device_rs = pipeline.get_active_profile().get_device()
advanced_mode = rs.rs400_advanced_mode(device_rs)

if not advanced_mode.is_enabled():
    advanced_mode.toggle_advanced_mode(True)
    time.sleep(2)
    device_rs = pipeline.get_active_profile().get_device()
    advanced_mode = rs.rs400_advanced_mode(device_rs)

with open(preset_file, 'r') as f:
    json_text = f.read()

advanced_mode.load_json(json_text)
print("[*] Loaded RealSense config from JSON")

align_to = rs.stream.color
align = rs.align(align_to)

pc = rs.pointcloud()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

save_dir = "pointcloud_capture"
os.makedirs(save_dir, exist_ok=True)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("Không đọc được frame")
        break

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(480, 640, 3)

    img = np.asanyarray(color_frame.get_data())
    img = cv2.resize(img, (640, 480))
    output = img.copy()

    results = model(img, verbose=False)
    captured_pc = None
    bb_coords = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half_size = 30
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(img.shape[1], cx + half_size)
            y2 = min(img.shape[0], cy + half_size)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0: continue
            h, w = roi.shape[:2]
            if h < 10 or w < 10: continue

            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

            lower_green = np.array([20, 60, 35])
            upper_green = np.array([95, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            center_mask = np.zeros((h, w), dtype=np.uint8)
            # ===== CHỈ SỬA DÒNG NÀY =====
            cv2.ellipse(center_mask, (w//2, h//2), (int(w*0.30), int(h*0.30)), 0, 0, 360, 255, -1)
            mask = cv2.bitwise_and(mask, center_mask)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)

            gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[mask > 0] = cv2.GC_PR_FGD

            inner_mask = np.zeros((h, w), dtype=np.uint8)
            # ===== CHỈ SỬA DÒNG NÀY =====
            cv2.ellipse(inner_mask, (w//2, h//2), (int(w*0.20), int(h*0.20)), 0, 0, 360, 255, -1)
            gc_mask[inner_mask > 0] = cv2.GC_FGD

            border = 5
            gc_mask[:border, :] = cv2.GC_BGD
            gc_mask[-border:, :] = cv2.GC_BGD
            gc_mask[:, :border] = cv2.GC_BGD
            gc_mask[:, -border:] = cv2.GC_BGD

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            try:
                cv2.grabCut(roi, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                continue

            final_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype('uint8')

            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, 1)
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 300:
                    c = cv2.convexHull(c)
                    c = c + np.array([[x1, y1]])
                    cv2.drawContours(output, [c], -1, (0, 0, 255), 1)

                    pc_mask = np.zeros((480, 640), dtype=np.uint8)
                    cv2.drawContours(pc_mask, [c], -1, 255, -1)

                    apple_pc_data = vtx[pc_mask == 255]
                    apple_pc_data = apple_pc_data[apple_pc_data[:, 2] > 0]

                    captured_pc = apple_pc_data
                    bb_coords = (x1, y1, x2, y2)

                    ys, xs = np.where(pc_mask == 255)
                    apple_pc_data = vtx[ys, xs]  # lấy toàn bộ điểm 3D trong contour
                    apple_pc_data = apple_pc_data[(apple_pc_data[:, 2] > 0.6) & (apple_pc_data[:, 2] <= 0.805)]

                    if len(apple_pc_data) > 0:
                        for pt in apple_pc_data:
                            u, v = int(pt[0]), int(pt[1])  # chỉ dùng để chiếu lên 2D nếu cần
                            # output[v, u] = img[v, u]  # hiển thị màu RGB thật từ RealSense


    cv2.imshow("Apple Contour", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('e') and captured_pc is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # ===== LƯU PLY toàn khung =====
        full_points = vtx.reshape(-1, 3)
        full_points = full_points[(full_points[:, 2] > 0.7) & (full_points[:, 2] <= 0.805)]
        ply_full_file = os.path.join(save_dir, f"PLY_full_{timestamp}.ply")
        with open(ply_full_file, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(full_points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in full_points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"[+] Saved: {ply_full_file}")

        # ===== LƯU PLY trong bounding box =====
        ply_bb_file = os.path.join(save_dir, f"PLY_in_BB_{timestamp}.ply")
        with open(ply_bb_file, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(captured_pc)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in captured_pc:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"[+] Saved: {ply_bb_file}")

        # ===== LƯU RGB toàn khung =====
        rgb_full_file = os.path.join(save_dir, f"RGB_full_{timestamp}.png")
        cv2.imwrite(rgb_full_file, img)
        print(f"[+] Saved: {rgb_full_file}")

        # ===== LƯU RGB trong bounding box =====
        x1, y1, x2, y2 = bb_coords
        rgb_bb = img[y1:y2, x1:x2]
        rgb_bb_file = os.path.join(save_dir, f"RGB_in_BB_{timestamp}.png")
        cv2.imwrite(rgb_bb_file, rgb_bb)
        print(f"[+] Saved: {rgb_bb_file}")

        # ===== LƯU depth toàn khung =====
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_full_file = os.path.join(save_dir, f"Depth_full_{timestamp}.png")
        cv2.imwrite(depth_full_file, depth_image)
        print(f"[+] Saved: {depth_full_file}")

        # ===== LƯU depth trong bounding box =====
        depth_bb = depth_image[y1:y2, x1:x2]
        depth_bb_file = os.path.join(save_dir, f"Depth_in_BB_{timestamp}.png")
        cv2.imwrite(depth_bb_file, depth_bb)
        print(f"[+] Saved: {depth_bb_file}")

    if key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()