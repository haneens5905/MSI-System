# MSI-System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
</p>

Automated waste material classification system using SVM and k-NN on image feature vectors, with real-time camera deployment.

---

## Project Overview

- Classifies waste material images into 7 categories using machine learning
- Implements a full ML pipeline: data augmentation, feature extraction, classifier training, and evaluation
- Trains and compares two classifiers: Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN)
- Handles out-of-distribution inputs via a rejection mechanism (Unknown class)
- Deploys the best-performing model in a live real-time camera application

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
| 6 | Unknown | Out-of-distribution or blurred inputs |

---

## Repository Structure

```
MSI-System/
│
├── data/
│   ├── raw/                        # Original unmodified dataset
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── augmented/                  # Augmented and balanced dataset (~500 per class)
│
├── features/
│   ├── X_train.npy                 # Training feature vectors
│   ├── X_val.npy                   # Validation feature vectors
│   ├── y_train.npy                 # Training labels
│   └── y_val.npy                   # Validation labels
│
├── models/
│   ├── svm_model.pkl               # Saved trained SVM model
│   └── knn_model.pkl               # Saved trained k-NN model
│
├── src/
│   ├── augmentation.py             # Data augmentation pipeline
│   ├── feature_extraction.py       # Image to feature vector conversion
│   ├── train_svm.py                # SVM training and evaluation
│   ├── train_knn.py                # k-NN training and evaluation
│   └── realtime_app.py             # Real-time camera classification app
│
├── report/
│   ├── project_brief.pdf           # Original project requirements
│   └── technical_report.pdf        # Final submitted technical report
│
├── requirements.txt                # Required Python packages
├── .gitignore
└── README.md
```

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Augment the Data
```bash
python src/augmentation.py
```

### Step 2 — Extract Features
```bash
python src/feature_extraction.py
```

### Step 3 — Train SVM
```bash
python src/train_svm.py
```

### Step 4 — Train k-NN
```bash
python src/train_knn.py
```

### Step 5 — Run Real-Time App
```bash
python src/realtime_app.py
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Image processing and real-time camera feed |
| scikit-learn | SVM and k-NN classifier implementation |
| NumPy | Feature vector storage and manipulation |
| scikit-image | Feature descriptor extraction |
| joblib | Model serialization and saving |
| Pillow | Image loading and augmentation |
