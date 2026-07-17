import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

save_id = 0

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

            if name == "apple":

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                expand = 20

                x1 = max(0, x1 - expand)
                y1 = max(0, y1 - expand)
                x2 = min(color.shape[1], x2 + expand)
                y2 = min(color.shape[0], y2 + expand)

                bbox = (x1, y1, x2, y2)

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

        save_id += 1
        filename = f"apple_{save_id}.ply"

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

    if key == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()