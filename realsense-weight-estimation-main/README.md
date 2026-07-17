# 📷 RealSense 3D Weight Estimation

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Intel RealSense](https://img.shields.io/badge/Intel-RealSense%20D400-0071C5.svg)

Estimating the weight of objects without direct physical contact is a highly valuable but challenging task, especially for livestock management and delicate precision agriculture. 

This project explores a non-invasive approach to weight estimation using depth data captured from an **Intel RealSense depth camera**. By leveraging 3D spatial information, point cloud processing, and machine learning techniques, the system approximates object weight with high accuracy while maintaining the flexibility needed for real-world deployment.

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download model files
Model weights are **not stored in this repository** (too large for Git). Download them and place them in the project root:

| File | Description | Link |
|---|---|---|
| `apple_weight.weights.h5` | Keras weight-estimation model | *(add your link here)* |
| `best.pt` | YOLO detection model | *(add your link here)* |
| `scaler.save` | Fitted sklearn scaler | *(add your link here)* |

> Tip: host these files on [Google Drive](https://drive.google.com), [Hugging Face](https://huggingface.co), or as [GitHub Release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

### 3. Connect your RealSense camera and run
```bash
python main.py
```

---

## 📂 Datasets

The raw data and processed datasets utilized in this project can be accessed via the following links:
* [Dataset Part 1](https://drive.google.com/drive/folders/1h24DYCF9H0KQb8R6RsqMh2ViEKZy9Y7W?usp=sharing)
* [Dataset Part 2](https://drive.google.com/drive/folders/1b4NnNTcvQ28SxFJmBKcWwM9m5h-CvECj?usp=sharing)
* [Dataset Part 3](https://drive.google.com/drive/folders/1Xx4uSQ50vD44oNztQ7BtBi-Ms4JzBG8L?usp=sharing)
* [Dataset Part 4](https://drive.google.com/drive/folders/1XRewnuMG_VzrsmF2ozqbEqQ6eZoxHgVg?usp=sharing)

---

## 🧠 Methodology & Pipeline

Unlike traditional methods that rely on physical scales, this approach extracts visual and spatial cues—such as volume, geometric shape, and surface structure—directly from 3D point clouds.

### Core Processing Pipeline
1. **Data Acquisition:** Capturing RGB and depth maps using Intel RealSense sensors.
2. **3D Reconstruction:** Converting depth maps into dense point cloud representations (`.ply`).
3. **Preprocessing & Filtering:** Isolating the target object and removing background noise using spatial filtering.
4. **Feature Extraction:** Extracting geometric features and utilizing deep learning (e.g., bounding box extraction, Unit Ball Normalization).
5. **Regression & Estimation:** Applying machine learning models to map 3D features to physical weight.

<div align="center">
  <img width="1245" alt="Pipeline Step 1" src="https://github.com/user-attachments/assets/b12dbf9e-5194-46e5-bec0-6e370f3fe1a5" />
  <img width="1239" alt="Pipeline Step 2" src="https://github.com/user-attachments/assets/e3b31442-5309-4ec9-be3d-b7b932627956" />
</div>

---

## 📊 Current Progress & Results

The foundational data processing pipeline is fully functional. The system has been successfully tested on diverse use cases, including **livestock (pigs)** and **agricultural products (strawberries, apples)**. Initial regression results demonstrate strong feasibility for depth-based weight prediction.

<div align="center">
  <img width="1220" alt="Result Visualization 1" src="https://github.com/user-attachments/assets/612400d1-e738-4c36-a8ee-f7e21ece9ea0" />
  <img width="1190" alt="Result Visualization 2" src="https://github.com/user-attachments/assets/51ded93c-6f9f-49f7-a1b8-0e0d10e8f577" />
  <img width="1234" alt="Result Visualization 3" src="https://github.com/user-attachments/assets/ba78adf0-1d4d-4773-8035-7045595d0102" />
  <img width="1163" alt="Result Visualization 4" src="https://github.com/user-attachments/assets/745e3943-f578-4d4d-afcb-51da4e27e2e9" />
  <img width="1110" alt="Result Visualization 5" src="https://github.com/user-attachments/assets/13856dbb-3f26-4871-9383-5e39dd25c7dc" />
</div>

---

## 🚀 Ongoing & Future Work

While the core system is operational, the project is under active development. Over the next 1–2 months, research will focus on:

* **Advanced Deep Learning Architectures:** Transitioning from traditional geometric extraction to native 3D deep learning models (e.g., PointNet, Point Transformer) for regression.
* **Multimodal Fusion:** Fusing high-resolution RGB visual data with depth sensor inputs to capture density and textural variations.
* **Environmental Robustness:** Improving real-world sensor calibration and stability under variable lighting and occlusions.

---

## 📚 Recommended Reading & References

For researchers and developers extending this work, a solid understanding of point cloud processing and depth-based volume estimation is highly recommended. Below are key academic references (IEEE/Elsevier) that heavily inform the techniques used in this repository:

### Livestock Weight Estimation
* [1] W. Wang *et al.*, "A non-contact pig weight estimation system based on 3D point cloud features," *IEEE Access*, vol. 7, pp. 12435-12445, 2019.
* [2] M. Kashiha *et al.*, "Automatic weight estimation of individual pigs using image analysis," *Computers and Electronics in Agriculture*, vol. 107, pp. 38-45, 2014.

### Agricultural Product Measurement
* [3] Y. Li *et al.*, "Non-destructive weight estimation of fruit based on 3D computer vision," *IEEE Transactions on Instrumentation and Measurement*, vol. 70, pp. 1-10, 2021.
* [4] Q. Wang *et al.*, "3D Point Cloud Processing for Crop Yield Prediction and Fruit Quality Assessment: A Review," *IEEE Internet of Things Journal*, 2022.

> **Note:** If you are building upon this pipeline, it is heavily advised to review the specific depth-matching algorithms and hardware constraints of the Intel RealSense D400 series outlined in the manufacturer's whitepapers.
