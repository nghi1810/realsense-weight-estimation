import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pyrealsense2 as rs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Device: {device}")

model = YOLO("apple.pt")
model.to(device)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

pc = rs.pointcloud()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            half_size = 30
            x1 = cx - half_size
            y1 = cy - half_size
            x2 = cx + half_size
            y2 = cy + half_size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

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

            lower_green = np.array([20, 60, 35])
            upper_green = np.array([95, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)

            center_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                center_mask,
                (w // 2, h // 2),
                (max(1, int(w * 0.40)), max(1, int(h * 0.40))),
                0, 0, 360, 255, -1
            )

            mask = cv2.bitwise_and(mask, center_mask)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

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

                    pc_mask = np.zeros((480, 640), dtype=np.uint8)
                    cv2.drawContours(pc_mask, [c], -1, 255, -1)
                    
                    apple_pc_data = vtx[pc_mask == 255]
                    apple_pc_data = apple_pc_data[apple_pc_data[:, 2] > 0]

                    ys, xs = np.where(pc_mask == 255)

                    valid = vtx[ys, xs, 2] > 0
                    ys = ys[valid]
                    xs = xs[valid]

                    output[ys, xs] = (255, 0, 0)

                    if len(apple_pc_data) > 0:
                        avg_z = np.mean(apple_pc_data[:, 2])
                        points_count = len(apple_pc_data)
                        
                        text = f"Z: {avg_z:.3f}m | Pts: {points_count}"
                        cv2.putText(output, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Apple Contour", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()