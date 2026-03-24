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
<img width="996" height="470" alt="image" src="https://github.com/user-attachments/assets/13c8f6e0-c282-4346-8f9a-3fd20c1fafd4" />

model pointnet

<img width="531" height="547" alt="image" src="https://github.com/user-attachments/assets/dadee175-42da-4b6e-94b3-4fd582384ee8" />

<img width="1172" height="613" alt="image" src="https://github.com/user-attachments/assets/060e1edc-cfb0-43ce-bf75-f45a225412d3" />


model DGCNN

<img width="1666" height="501" alt="image" src="https://github.com/user-attachments/assets/cf296b5e-d9f7-4e37-ba2c-cf1844288284" />


<img width="569" height="591" alt="image" src="https://github.com/user-attachments/assets/cd3eb1e5-b9bb-45cf-80c8-328e59996c0b" />



