import cv2
import os
import numpy as np
from collections import Counter

# Constants and settings
orb = cv2.ORB_create(nfeatures=1000)
data_dir = 'data'
classes = ['R10', 'R20', 'R50', 'R100', 'R200']
k = 23  # Number of nearest neighbors to consider for the final classification

# Function to preprocess images
def preprocess_image(img):
    img_blurred = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_resized = cv2.resize(img_eq, (574, 265))
    # Adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img_thresh

# Function to extract features using ORB
def extract_features(img):
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return descriptors

# Function to load training data
def load_training_data(data_dir, classes):
    train_descriptors = []
    train_labels = []
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(class_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                preprocessed_img = preprocess_image(img)
                descriptors = extract_features(preprocessed_img)
                if descriptors is not None:
                    train_descriptors.append(descriptors)
                    train_labels.append(label)
    return train_descriptors, train_labels

# Function to classify an image based on extracted features
def classify_image(test_descriptors, train_descriptors, train_labels, k=23):
    bf = cv2.BFMatcher()
    all_matches = []
    for i, train_desc in enumerate(train_descriptors):
        matches = bf.knnMatch(test_descriptors, train_desc, k=2)
        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        all_matches.extend([(train_labels[i], m) for m in good_matches])
    
    all_matches.sort(key=lambda x: x[1].distance)
    top_k_matches = all_matches[:k]

    matched_labels = [label for label, match in top_k_matches]
    most_common_label = Counter(matched_labels).most_common(1)[0][0]
    return most_common_label

# Function to run the live webcam feed and classify images
def live_classification():
    train_descriptors, train_labels = load_training_data(data_dir, classes)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_img = preprocess_image(frame)
        test_descriptors = extract_features(preprocessed_img)
        
        if test_descriptors is not None:
            predicted_class = classify_image(test_descriptors, train_descriptors, train_labels, k)
            cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_classification()