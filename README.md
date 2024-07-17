## Image Processing Pipeline Nodes using OpenCV

---

## Description

This project demonstrates various image processing techniques using OpenCV and the Dataloop platform. It includes functionalities for contrast enhancement, image cropping, annotation management, and face blurring, each implemented as a node in the pipeline.

## Nodes

#### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Enhances the contrast of grayscale images using CLAHE, which is particularly useful for improving the visibility of features in images with varying illumination.

#### 2. Crop Images

Crops images based on bounding box annotations and uploads the cropped images to the same dataset, preserving annotation details and metadata.

#### 3. Blur Faces

Detects faces in images using a pre-trained Haar Cascade classifier and applies a blur effect to the detected face regions, helping to anonymize individuals in the images.
