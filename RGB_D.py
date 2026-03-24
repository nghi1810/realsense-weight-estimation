import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os

# load model YOLOv8
model = YOLO("yolov8n.pt")

# tạo thư mục lưu
os.makedirs("dataset/rgb", exist_ok=True)
os.makedirs("dataset/depth", exist_ok=True)

save_id = 0

# RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

print("Press E to save apple RGB + depth npy")

while True:

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame:
        continue

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    # YOLO detect
    results = model(color, conf=0.5)

    boxes = results[0].boxes

    for box in boxes:

        cls = int(box.cls[0])

        # class 47 = apple (COCO)
        if cls != 47:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])


        # crop RGB
        roi_rgb = color[y1:y2, x1:x2]

        # crop depth
        roi_depth = depth[y1:y2, x1:x2]

        cv2.imshow("apple rgb", roi_rgb)

    cv2.imshow("RGB", color)

    key = cv2.waitKey(1)

    if key == ord('e'):

        if 'roi_rgb' in locals():

            rgb_path = f"dataset/rgb/apple_{save_id}.png"
            depth_path = f"dataset/depth/apple_{save_id}.npy"

            cv2.imwrite(rgb_path, roi_rgb)
            np.save(depth_path, roi_depth)

            print("Saved:", rgb_path, depth_path)

            save_id += 1

    if key == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()