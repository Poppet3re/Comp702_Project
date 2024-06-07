# Bank Note Classification

## Introduction
Banknote recognition is a highly relevant task in the modern era, and it has applications in financial transactions and security systems. Identifying and classifying banknotes is highly useful for ensuring the integrity and efficiency of financial systems. This project aims to develop an image-processing system capable of classifying South African banknotes, including old and new notes, invariant to sides, size, and orientation of the banknote. 

This project will explore multiple image processing techniques and compare their results to develop an effective system for banknote classification. We will divide this project into four stages: Image Processing and Enhancement, Segmentation, Feature Extraction, and Classification. Each stage will involve implementing and comparing various algorithms, primarily utilizing the existing OpenCV library in Python, focusing on performance and efficacy. We aim to shed insight into various techniques that can be applied to the image processing of banknotes.

## Data Collection and Labeling
The first step in the project is to collect a dataset of South African banknotes. The dataset will include images of both old and new notes, captured from different angles and under different lighting conditions. The dataset will be labeled with the denomination of each banknote, allowing us to train and evaluate the classification model.

We will use a combination of real-world images and synthetic data to create a diverse and representative dataset. The real-world images will be captured using a flatbed scanner or a high-resolution camera, as well as photographs taken from internet sources. We augment the dataset with synthetic data generated using image processing techniques to increase the diversity and size of the dataset. We consider the use of data augmentation techniques such as rotation, scaling, and flipping to create additional training samples as well as to improve the robustness of the model.

## Image Processing and Enhancement
The first stage of the project involves preprocessing the banknote images to enhance their quality and make them suitable for further processing. This stage will include techniques such as image resizing, noise reduction, and contrast enhancement. The goal is to improve the quality of the images and make them more suitable for segmentation and feature extraction.

We aim to preserve important features of the banknotes while removing irrelevant information that may hinder the classification process. We will explore various image processing techniques, such as histogram equalization, adaptive thresholding, and edge detection, to enhance the banknote images. The performance of these techniques will be evaluated based on their ability to improve the quality of the images and make them more suitable for segmentation.

### Gray Scaling
The first step in image processing is to convert the banknote images to grayscale. While the color information may be useful in some cases, grayscale images are generally easier to process and analyze. We find that the feature extraction and classification algorithms often perform better on grayscale images, as they are less affected by variations in color.

### Noise Reduction
Noise reduction is an essential step in image processing, as it helps to remove unwanted artifacts and improve the quality of the images. We will explore various noise reduction techniques, such as Gaussian blurring and median filtering, to remove noise from the banknote images. The goal is to preserve important features of the banknotes while removing irrelevant information that may hinder the classification process.

### Contrast Enhancement
Contrast enhancement is another critical step in image processing, as it helps to improve the visibility of important features in the images. We will explore various contrast enhancement techniques, such as histogram equalization and adaptive histogram equalization, to enhance the banknote images. The goal is to improve the quality of the images and make them more suitable for segmentation and feature extraction.

## Segmentation
The second stage of the project involves segmenting the banknote images to extract the regions of interest. This stage will include techniques such as thresholding, contour detection, and region-based segmentation. The goal is to separate the banknote from the background and extract the features that are relevant for classification.

We primarily consider contour detection and the grab-cut algorithm for segmenting the banknote images. Contour detection is a simple and effective technique for extracting the outline of the banknote, while the grab-cut algorithm is a more sophisticated method that can separate the banknote from the background based on user input. We will compare the performance of these techniques based on their ability to accurately segment the banknote images.

We found that the performance of the segmentation algorithms is highly dependent on the quality of the preprocessed images. Therefore, we aim to optimize the image processing techniques to improve the accuracy of the segmentation process.

The grab-cut algorithm was noted to be significantly computationally expensive, and the performance was relatively poor due to the fixed starting rectangle. This limited the algorithm's ability to accurately segment the banknote images, especially when the banknotes were fittingly placed in the image.

Therefore we decided to focus on contour detection as the primary segmentation technique, as it was more computationally efficient and provided satisfactory results for our dataset.

## Feature Extraction
The third stage of the project involves extracting features from the segmented banknote images. This stage will include techniques such as shape analysis, texture analysis, and feature encoding. The goal is to represent the banknote images in a way that captures the relevant information for classification as feature vectors that can be used by the classification model.

We primarily consider builtin OpenCV functions for feature extraction, such as Hu Moments and Haralick Texture features, as well as ORB (Oriented FAST and Rotated BRIEF) and SIFT (Scale-Invariant Feature Transform) for feature detection and description. We aim to compare the performance of these techniques based on their ability to capture the relevant information from the banknote images.

Since all of these functions extract features to variable lengths, we cluster the features using KMeans clustering with 1000 clusters to create a fixed-length feature vector for each banknote image. This fixed-length feature vector can then be used as input to the classification model.

## Classification
The final stage of the project involves training and evaluating a classification model on the extracted features. This stage will include techniques such as Support Vector Machines (SVM), Random Forest, and K-Nearest Neighbors (KNN).

Since we have our data preprocessed, segmented, and features extracted, we can now train a classification model to predict the denomination of the banknotes. We split the dataset into training and testing sets, and we evaluate the performance of the classification model based on metrics such as accuracy, precision, recall, and F1 score.