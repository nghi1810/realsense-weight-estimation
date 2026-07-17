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
preset_file = r"C:\Users\phand\Desktop\new\settng15.json"

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

save_dir = "pointcloud_capture1"
os.makedirs(save_dir, exist_ok=True)

# ===== TẠO FOLDER APPLE_1 BAN ĐẦU =====
apple_idx = 1
apple_folder = f"apple_{apple_idx}"
apple_save_dir = os.path.join(save_dir, apple_folder)
os.makedirs(apple_save_dir, exist_ok=True)

# Tạo các thư mục con bên trong apple_folder
sub_dirs = [
    "RGB_full", "RGB_in_BB",
    "depth_full", "depth_in_BB",
    "PLY_full", "PLY_in_BB", "PLY_in_contour",
    "npy_full", "npy_in_BB", "npy_in_contour"
]
for sub in sub_dirs:
    os.makedirs(os.path.join(apple_save_dir, sub), exist_ok=True)
# =====================================================================

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
            half_size = 50
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(img.shape[1], cx + half_size)
            y2 = min(img.shape[0], cy + half_size)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            h, w = roi.shape[:2]
            if h < 10 or w < 10:
                continue

            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

            # ✅ Siết lại màu xanh (loại bỏ vùng tối + bóng)
            lower_green = np.array([35, 110, 60])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            center_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(center_mask, (w//2, h//2), (int(w*0.30), int(h*0.30)), 0, 0, 360, 255, -1)
            mask = cv2.bitwise_and(mask, center_mask)

            # ✅ Kernel nhỏ lại để không ăn vào vùng bóng
            kernel = np.ones((3, 3), np.uint8)

            # ✅ Ưu tiên OPEN để xoá chấm đen trước
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

            gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[mask > 0] = cv2.GC_PR_FGD

            inner_mask = np.zeros((h, w), dtype=np.uint8)
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

            # ✅ Làm mịn nhưng không tạo bóng
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, 1)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, 1)

            # ❌ bỏ blur để tránh viền bị "bóng"
            # final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

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
                    apple_pc_data = vtx[ys, xs]
                    apple_pc_data = apple_pc_data[(apple_pc_data[:, 2] > 0.6) & (apple_pc_data[:, 2] <= 0.77)]

    cv2.imshow("Apple Contour", output)
    key = cv2.waitKey(1) & 0xFF

    # ===== NHẤN 'A' ĐỂ TĂNG SỐ APPLE =====
    if key == ord('a') or key == ord('A'):
        apple_idx += 1
        apple_folder = f"apple_{apple_idx}"
        apple_save_dir = os.path.join(save_dir, apple_folder)
        os.makedirs(apple_save_dir, exist_ok=True)
        for sub in sub_dirs:
            os.makedirs(os.path.join(apple_save_dir, sub), exist_ok=True)
        print(f"[*] Switched to {apple_folder}")


    # ===== NHẤN 'E' ĐỂ LƯU =====
    if key == ord('e'):
        # Tạo số thứ tự file cho apple hiện tại
        def get_next_file_idx(folder, prefix):
            existing = [f for f in os.listdir(folder) if f.startswith(prefix)]
            return len(existing) + 1

        # Depth thực tế từ RealSense
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_visual = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # ===== LƯU RGB full =====
        idx = get_next_file_idx(os.path.join(apple_save_dir, "RGB_full"), "RGB_full_")
        cv2.imwrite(os.path.join(apple_save_dir, "RGB_full", f"RGB_full_{idx:03d}.png"), img)

        # ===== LƯU RGB BB =====
        try:
            idx = get_next_file_idx(os.path.join(apple_save_dir, "RGB_in_BB"), "RGB_in_BB_")
            rgb_bb = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(apple_save_dir, "RGB_in_BB", f"RGB_in_BB_{idx:03d}.png"), rgb_bb)
        except NameError:
            pass

        # ===== LƯU depth full =====
        idx = get_next_file_idx(os.path.join(apple_save_dir, "depth_full"), "Depth_full_")
        cv2.imwrite(os.path.join(apple_save_dir, "depth_full", f"Depth_full_{idx:03d}.png"), depth_image_visual)

        # ===== LƯU depth BB =====
        try:
            idx = get_next_file_idx(os.path.join(apple_save_dir, "depth_in_BB"), "Depth_in_BB_")
            depth_bb_visual = depth_image_visual[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(apple_save_dir, "depth_in_BB", f"Depth_in_BB_{idx:03d}.png"), depth_bb_visual)
        except NameError:
            pass

        # ===== LƯU PLY full =====
        full_points = vtx.reshape(-1, 3)
        full_points = full_points[full_points[:, 2] > 0]
        idx = get_next_file_idx(os.path.join(apple_save_dir, "PLY_full"), "PLY_full_")
        ply_full_file = os.path.join(apple_save_dir, "PLY_full", f"PLY_full_{idx:03d}.ply")
        with open(ply_full_file, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(full_points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in full_points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # ===== LƯU PLY BB =====
        try:
            bb_points = vtx[y1:y2, x1:x2].reshape(-1, 3)
            bb_points = bb_points[bb_points[:, 2] > 0]
            idx = get_next_file_idx(os.path.join(apple_save_dir, "PLY_in_BB"), "PLY_in_BB_")
            ply_bb_file = os.path.join(apple_save_dir, "PLY_in_BB", f"PLY_in_BB_{idx:03d}.ply")
            with open(ply_bb_file, 'w') as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(bb_points)}\n")
                f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                for p in bb_points:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")
        except NameError:
            pass

        # ===== LƯU PLY contour với Z filter =====
        try:
            if 'contours' in locals() and len(contours) > 0:
                contour_points = []
                for yy, xx in zip(*np.where(final_mask > 0)):
                    pt = vtx[y1+yy, x1+xx]
                    if 0.6 < pt[2] <= 0.67:
                        contour_points.append(pt)
                contour_points = np.array(contour_points)
                idx = get_next_file_idx(os.path.join(apple_save_dir, "PLY_in_contour"), "PLY_in_contour_")
                ply_contour_file = os.path.join(apple_save_dir, "PLY_in_contour", f"PLY_in_contour_{idx:03d}.ply")
                with open(ply_contour_file, 'w') as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(contour_points)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                    for p in contour_points:
                        f.write(f"{p[0]} {p[1]} {p[2]}\n")
        except NameError:
            pass

        # ===== LƯU npy full =====
        idx = get_next_file_idx(os.path.join(apple_save_dir, "npy_full"), "full_")
        np.save(os.path.join(apple_save_dir, "npy_full", f"full_{idx:03d}.npy"), full_points)

        # ===== LƯU npy BB =====
        try:
            idx = get_next_file_idx(os.path.join(apple_save_dir, "npy_in_BB"), "in_BB_")
            np.save(os.path.join(apple_save_dir, "npy_in_BB", f"in_BB_{idx:03d}.npy"), bb_points)
        except NameError:
            pass

        # ===== LƯU npy contour =====
        try:
            if 'contours' in locals() and len(contours) > 0:
                contour_points_npy = np.array([p for p in contour_points if 0.6 < p[2] <= 0.67])
                idx = get_next_file_idx(os.path.join(apple_save_dir, "npy_in_contour"), "in_contour_")
                np.save(os.path.join(apple_save_dir, "npy_in_contour", f"in_contour_{idx:03d}.npy"), contour_points_npy)
        except NameError:
            pass

        print(f"[+] Saved all data for {apple_folder} (index {idx:03d})")




    # ===== NHẤN 'Q' ĐỂ THOÁT =====
    if key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()