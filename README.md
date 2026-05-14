# realsense-weight-estimation
dataset
https://drive.google.com/drive/folders/1h24DYCF9H0KQb8R6RsqMh2ViEKZy9Y7W?usp=sharing
https://drive.google.com/drive/folders/1b4NnNTcvQ28SxFJmBKcWwM9m5h-CvECj?usp=sharing
https://drive.google.com/drive/folders/1Xx4uSQ50vD44oNztQ7BtBi-Ms4JzBG8L?usp=sharing
https://drive.google.com/drive/folders/1XRewnuMG_VzrsmF2ozqbEqQ6eZoxHgVg?usp=sharing

Introduction

In recent years, the application of computer vision and depth sensing technologies has significantly expanded in agriculture and livestock management. Estimating the weight of objects without direct physical contact is a challenging yet highly valuable task, especially in scenarios involving animals such as pigs or delicate agricultural products like strawberries.

This project explores a non-invasive approach to weight estimation using depth data captured from an Intel RealSense camera. By leveraging 3D spatial information, point cloud processing, and machine learning techniques, the system aims to approximate object weight with reasonable accuracy while maintaining scalability and flexibility for real-world deployment.

Project Description

The core idea of this project is to utilize depth images and point cloud data to extract meaningful geometric features that correlate with the physical weight of an object. Unlike traditional methods that rely on direct measurement, this approach focuses on visual and spatial cues such as volume, shape, and surface structure.

The implemented pipeline includes:

Capturing depth data using RealSense sensors
Converting depth maps into point cloud representations
Processing and filtering 3D data
Extracting features related to object geometry
Applying estimation techniques to predict weight

The system has been tested on use cases involving livestock (e.g., pigs) and agricultural products (e.g., strawberries), where non-contact measurement is particularly beneficial.
<img width="1245" height="615" alt="Screenshot 2026-05-14 at 11 30 43" src="https://github.com/user-attachments/assets/b12dbf9e-5194-46e5-bec0-6e370f3fe1a5" />
<img width="1239" height="700" alt="Screenshot 2026-05-14 at 11 30 48" src="https://github.com/user-attachments/assets/e3b31442-5309-4ec9-be3d-b7b932627956" />
Current Progress

At the current stage, the core system has been successfully implemented. The data processing pipeline is functional, and initial weight estimation results have been obtained. These results demonstrate the feasibility of using depth-based approaches for this problem.

However, the project is still under active development. While the foundational components are complete, further improvements are required to enhance accuracy, robustness, and generalization across different object types and environmental conditions.

Ongoing and Future Work

Over the next one to two months, the project will continue to evolve with a focus on both research and practical improvements. Key directions include:

Exploring more advanced machine learning and deep learning models
Investigating multimodal fusion techniques, combining RGB images, depth data, and potentially other sensor inputs
Improving feature extraction from point clouds and 3D representations
Enhancing system stability in real-world scenarios
Conducting more extensive experiments and evaluations

These efforts aim to transform the current prototype into a more reliable and scalable solution.

Important Note for Readers

One of the most important recommendations for anyone interested in this project is to invest time in studying related research. In particular, you are strongly encouraged to read academic papers and technical reports on:

RealSense camera applications
Point cloud processing and 3D computer vision
Depth-based measurement systems
Weight estimation methods for livestock (especially pigs)
Fruit weight estimation (e.g., strawberries and similar crops)

A solid understanding of these topics will provide valuable context and significantly improve your ability to work with and extend this project.

Conclusion

This project represents an ongoing effort to bridge the gap between computer vision and real-world measurement tasks. By utilizing depth sensing and 3D data, it demonstrates a promising direction for non-contact weight estimation in agriculture and beyond.

Although still under development, the current system lays a strong foundation for future research and practical applications. Continuous improvements and integration of advanced techniques such as multimodal learning are expected to further enhance its performance and usability.

Estimate object weight using Intel RealSense depth camera and computer vision techniques.
<img width="1220" height="663" alt="Screenshot 2026-05-14 at 11 30 54" src="https://github.com/user-attachments/assets/612400d1-e738-4c36-a8ee-f7e21ece9ea0" />

<img width="1190" height="567" alt="Screenshot 2026-05-14 at 11 30 20" src="https://github.com/user-attachments/assets/51ded93c-6f9f-49f7-a1b8-0e0d10e8f577" />
<img width="1234" height="686" alt="Screenshot 2026-05-14 at 11 30 25" src="https://github.com/user-attachments/assets/ba78adf0-1d4d-4773-8035-7045595d0102" />
<img width="1163" height="644" alt="Screenshot 2026-05-14 at 11 30 32" src="https://github.com/user-attachments/assets/745e3943-f578-4d4d-afcb-51da4e27e2e9" />
<img width="1110" height="561" alt="Screenshot 2026-05-14 at 11 30 37" src="https://github.com/user-attachments/assets/13856dbb-3f26-4871-9383-5e39dd25c7dc" />





