import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pyrealsense2 as rs

# ===== 1. LOAD YOLO =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Device: {device}")

model = YOLO("apple.pt")
model.to(device)

# ===== 2. VIDEO SOURCE (REALSENSE) =====
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ===== 3. CLAHE =====
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ===== 4. MAIN LOOP =====
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        print("Không đọc được frame")
        break

    img = np.asanyarray(color_frame.get_data())
    img = cv2.resize(img, (640, 480))
    output = img.copy()

    # ===== YOLO DETECT =====
    results = model(img, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ===== TẠO BOUNDING BOX CỐ ĐỊNH 100x100 =====
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            half_size = 30
            x1 = cx - half_size
            y1 = cy - half_size
            x2 = cx + half_size
            y2 = cy + half_size

            # Giới hạn trong frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            # ===== VẼ BOUNDING BOX =====
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            h, w = roi.shape[:2]
            if h < 10 or w < 10:
                continue

            # ===== TIỀN XỬ LÝ =====
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

            # ===== MASK MÀU XANH =====
            lower_green = np.array([20, 60, 35])
            upper_green = np.array([95, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)

            # ===== CENTER MASK =====
            center_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                center_mask,
                (w // 2, h // 2),
                (max(1, int(w * 0.40)), max(1, int(h * 0.40))),
                0, 0, 360, 255, -1
            )

            mask = cv2.bitwise_and(mask, center_mask)

            # ===== MORPHOLOGY =====
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # ===== GRABCUT =====
            gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[mask > 0] = cv2.GC_PR_FGD

            inner_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                inner_mask,
                (w // 2, h // 2),
                (max(1, int(w * 0.25)), max(1, int(h * 0.25))),
                0, 0, 360, 255, -1
            )
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

            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

            # ===== CONTOUR =====
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if area > 300:
                    perimeter = cv2.arcLength(c, True)
                    epsilon = 0.006 * perimeter
                    c = cv2.approxPolyDP(c, epsilon, True)
                    c = cv2.convexHull(c)

                    c = c + np.array([[x1, y1]])

                    cv2.drawContours(output, [c], -1, (0, 0, 255), 1)

    cv2.imshow("Apple Contour", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()