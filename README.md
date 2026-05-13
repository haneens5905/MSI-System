# MSI-System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--image-3E3E3E?style=flat-square&logo=python&logoColor=white" alt="scikit-image" />
  <img src="https://img.shields.io/badge/seaborn-4C72B0?style=flat-square&logo=python&logoColor=white" alt="seaborn" />
</p>

Automated waste material classification system using SVM and k-NN on image feature vectors, with real-time camera deployment.

---

## Overview

This system implements a full end-to-end machine learning pipeline for identifying waste materials from images. Raw images are converted into numerical feature vectors using handcrafted descriptors, two classifiers are trained and compared, and the best-performing model is deployed in a live camera application that classifies materials in real time.

---

## Material Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Glass | Bottles, jars |
| 1 | Paper | Newspapers, office paper |
| 2 | Cardboard | Boxes, packaging |
| 3 | Plastic | Water bottles, film |
| 4 | Metal | Aluminum cans, steel |
| 5 | Trash | Non-recyclable or contaminated waste |
| 6 | Unknown | Out-of-distribution or low-confidence inputs |

---

## Pipeline

```
Raw Images
    ↓ Data Augmentation       — balance all classes to 500 images each
    ↓ Feature Extraction      — HOG + HSV Histogram + LBP → 2302-dim vector
    ↓ Classifier Training     — SVM (71%) and k-NN (55%)
    ↓ Real-Time Deployment    — live camera app using best model (SVM)
```

---

## Results

| Model | Best Configuration | Validation Accuracy |
|-------|-------------------|-------------------|
| SVM | C=5, RBF kernel | 71.00% |
| k-NN | k=11, distance weighting | 55.00% |

SVM was selected as the deployment model based on its superior accuracy and faster prediction time.

<img src="models/svm_confusion_matrix.png" width="600"/>
<img src="models/knn_confusion_matrix.png" width="600"/>

### k-NN Hyperparameter Sweep

<img src="models/knn_experiment_results.png" width="600"/>

---

## Real-Time Demo

> Screenshot coming soon.

The live camera application captures frames, extracts features, and displays the predicted material class with confidence percentage in real time. Press **Q** to quit.

---

## Repository Structure

```
MSI-System/
│
├── data/
│   ├── raw/                        # original unmodified dataset
│   └── augmented/                  # augmented and balanced dataset (~500 per class)
│
├── features/
│   ├── X_train.npy                 # training feature vectors
│   ├── X_val.npy                   # validation feature vectors
│   ├── y_train.npy                 # training labels
│   ├── y_val.npy                   # validation labels
│   └── scaler.pkl                  # fitted standard scaler
│
├── models/
│   ├── knn_model.pkl               # trained k-NN model
│   ├── svm_model.pkl               # trained SVM model (see download link below)
│   ├── svm_confusion_matrix.png    # SVM confusion matrix
│   ├── knn_confusion_matrix.png    # k-NN confusion matrix
│   ├── knn_experiment_results.png  # k-NN accuracy vs k plot
│   └── knn_classification_report.txt
│
├── src/
│   ├── augmentation.py             # data augmentation pipeline
│   ├── feature_extraction.py       # image to feature vector conversion
│   ├── train_svm.py                # SVM training and evaluation
│   ├── train_knn.py                # k-NN training and evaluation
│   └── realtime_app.py             # real-time camera classification app
│
├── report/
│   └── MSI_Report.docx             # technical report
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the SVM Model
The trained SVM model is too large for GitHub. Download it and place it in `models/`:

[Download svm_model.pkl from Google Drive](https://drive.google.com/file/d/12YobzdaWK0lZNoNTBtPQbYGBBOTKWwcu/view?usp=sharing)

---

## How to Run

```bash
# step 1 — augment and balance the dataset
python src/augmentation.py

# step 2 — extract feature vectors from all images
python src/feature_extraction.py

# step 3 — train and evaluate SVM
python src/train_svm.py

# step 4 — train and evaluate k-NN
python src/train_knn.py

# step 5 — run the real-time camera app
python src/realtime_app.py
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Image processing and real-time camera feed |
| scikit-learn | SVM and k-NN implementation |
| scikit-image | HOG and LBP feature extraction |
| NumPy | Feature vector storage and manipulation |
| joblib | Model serialization |
| Pillow | Image loading and augmentation |
| seaborn | Confusion matrix visualization |
| tqdm | Progress bars during training and extraction |
