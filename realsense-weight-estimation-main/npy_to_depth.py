import numpy as np
import cv2
import os

INPUT_FOLDER = r"C:\Users\phand\Desktop\New folder\npy\apple_2"   # folder chứa npy
OUTPUT_FOLDER = "depth_dataset"
IMG_SIZE = 224

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".npy")]

for file in files:

    INPUT_NPY = os.path.join(INPUT_FOLDER, file)

    # ==========================
    # load point cloud
    # ==========================

    points = np.load(INPUT_NPY)

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    px = (x_norm * (IMG_SIZE-1)).astype(int)
    py = (y_norm * (IMG_SIZE-1)).astype(int)

    depth_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for i in range(len(points)):

        xi = px[i]
        yi = py[i]
        d = z[i]

        if depth_img[yi, xi] == 0:
            depth_img[yi, xi] = d
        else:
            depth_img[yi, xi] = min(depth_img[yi, xi], d)

    # normalize only non-zero pixels
    mask = depth_img > 0

    z_min = depth_img[mask].min()
    z_max = depth_img[mask].max()

    depth_norm = np.zeros_like(depth_img)
    depth_norm[mask] = (depth_img[mask] - z_min) / (z_max - z_min)

    gray = np.zeros_like(depth_norm)
    gray[mask] = (1 - depth_norm[mask]) * 255

    gray = gray.astype(np.uint8)

    # ==========================
    # save
    # ==========================

    name = os.path.splitext(file)[0]

    png_path = os.path.join(OUTPUT_FOLDER, name + ".png")
    npy_path = os.path.join(OUTPUT_FOLDER, name + ".npy")

    cv2.imwrite(png_path, gray)
    np.save(npy_path, gray)

    print("Saved:", png_path)

print("Done.")