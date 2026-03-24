import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Folder lưu ảnh
os.makedirs("capture", exist_ok=True)
img_id = 0

print("Press E to save image")

try:
    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        # YOLO detect
        results = model(color, imgsz=1280, conf=0.5)

        for r in results:

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()

            if len(boxes) == 0:
                continue

            x1, y1, x2, y2 = map(int, boxes[0])

            # Crop RGB
            rgb_crop = color[y1:y2, x1:x2]

            # Crop depth
            depth_crop = depth[y1:y2, x1:x2]

            # Convert depth → grayscale
            depth_gray = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX)
            depth_gray = depth_gray.astype(np.uint8)

            # Phóng to để dễ nhìn
            rgb_show = cv2.resize(rgb_crop, (500, 500))
            depth_show = cv2.resize(depth_gray, (500, 500))

            cv2.imshow("RGB Apple", rgb_show)
            cv2.imshow("Depth Apple", depth_show)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        if key == ord('e'):  # Save
            rgb_name = f"capture/rgb_{img_id}.png"
            depth_name = f"capture/depth_{img_id}.png"

            cv2.imwrite(rgb_name, rgb_crop)
            cv2.imwrite(depth_name, depth_gray)

            print("Saved:", img_id)

            img_id += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()