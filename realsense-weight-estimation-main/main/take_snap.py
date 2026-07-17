import cv2
import os
import random
import torch
from ultralytics import YOLO

# ===== CONFIG =====
VIDEO_SOURCE = 0
OUTPUT_DIR = "dataset"
TRAIN_RATIO = 0.8
IMG_SIZE = 640

SPORT_BALL_CLASS = 32  # COCO
BOX_SIZE = 60         # 🔥 fixed bbox = 100x100

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("👉 Using device:", device)

# ===== CREATE FOLDER =====
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# ===== LOAD MODEL =====
model = YOLO("yolov8n.pt")
model.to(device)

# ===== VIDEO =====
cap = cv2.VideoCapture(VIDEO_SOURCE)

img_count = 0

print("👉 Nhấn 'E' để lưu ảnh, 'ESC' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    results = model(frame, imgsz=IMG_SIZE, conf=0.3, device=device)

    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])

            if cls != SPORT_BALL_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ===== LẤY CENTER =====
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ===== FIXED BBOX 100x100 =====
            half = BOX_SIZE // 2

            x1_new = max(0, cx - half)
            y1_new = max(0, cy - half)
            x2_new = min(w, cx + half)
            y2_new = min(h, cy + half)

            detections.append((x1_new, y1_new, x2_new, y2_new))

           

    cv2.imshow("Detect", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('e'):
        if len(detections) == 0:
            print("⚠️ Không có ball -> không lưu")
            continue

        x1, y1, x2, y2 = detections[0]

        # ===== YOLO FORMAT =====
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        split = "train" if random.random() < TRAIN_RATIO else "val"

        img_name = f"img_{img_count:05d}.jpg"
        label_name = f"img_{img_count:05d}.txt"

        img_path = f"{OUTPUT_DIR}/images/{split}/{img_name}"
        label_path = f"{OUTPUT_DIR}/labels/{split}/{label_name}"

        cv2.imwrite(img_path, frame)

        with open(label_path, "w") as f:
            f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

        print(f"✅ Saved: {img_name}")
        img_count += 1

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Done. Total images:", img_count)