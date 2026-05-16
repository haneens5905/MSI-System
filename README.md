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

## Project Overview

- Converts raw waste material images into fixed-length numerical feature vectors using **HOG**, **Color Histograms (BGR + HSV + LAB)**, **Color Moments**, **LBP**, **GLCM**, **Hu Moments**, **Edge Features**, and **Gabor Filters**
- Trains and compares two classifiers: **Support Vector Machine (SVM)** and **k-Nearest Neighbors (k-NN)**
- Implements a **confidence-based rejection mechanism** to handle unknown or ambiguous inputs
- Deploys the best-performing model in a **live real-time camera application**

---

## Dataset

The dataset contains images across six material classes organized into separate folders. Images vary in size, lighting, and orientation — reflecting real-world capture conditions.

| Class | Original Count |
|-------|---------------|
| Cardboard | 259 |
| Glass | 401 |
| Metal | 328 |
| Paper | 476 |
| Plastic | 386 |
| Trash | 110 |
| **Total** | **1960** |

The dataset is included in `data/raw/` and is augmented to 500 images per class (3000 total) before training.

---

## How It Works

### 1. Data Augmentation

Each class is augmented to exactly **500 images** using a combination of random transforms applied independently per image:

| Technique | Probability | Purpose |
|-----------|-------------|---------|
| Rotation (90°) | 50% | Handles arbitrary object orientation |
| Horizontal Flip | 50% | Simulates mirrored captures |
| Vertical Flip | 30% | Additional orientation variation |
| Color Jitter | 70% | Simulates varying lighting conditions |
| Gaussian Noise | 30% | Mimics sensor noise in cameras |
| Affine Transform | 60% | Simulates perspective and scale changes |

The dataset grows from **1960 to 3000 images — a 53% increase**. Images are then split 80/20 into training (2400) and validation (600) sets.

---

### 2. Feature Extraction

Each image is resized to **128×128 pixels** then converted into a **738-dimensional feature vector** by concatenating eight descriptors:

| Descriptor | What it captures | Details |
|------------|-----------------|---------|
| HOG | Shape and edge structure | 32×32 pixels per cell, 9 orientations |
| Color Histograms | Color distribution in BGR, HSV, and LAB | 32 bins per channel × 9 channels |
| Color Moments | Mean, std, median per HSV channel | Compact color summary |
| LBP | Surface texture | Uniform LBP, radius=3, 24 points |
| GLCM | Texture properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM) | 3 distances × 4 angles |
| Hu Moments | Shape invariant to rotation, scale, and translation | Log-transformed |
| Edge Features | Edge density and contour statistics via Canny | 4 values |
| Gabor Filters | Texture at 4 orientations | Mean and std per filter |

All feature vectors are standardized using **StandardScaler** (fitted on training data only) before being passed to the classifiers.

---

### 3. Classifiers

**SVM** — finds the maximum-margin decision boundary in feature space.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Kernel | RBF | Handles non-linear class boundaries |
| C | 5 (best from search over 1, 5, 10, 50, 100, 200, 500) | Optimal regularization |
| Gamma | scale | Auto-scales based on feature variance |
| Class weight | balanced | Handles remaining class imbalance |

**k-NN** — classifies by weighted vote among the k nearest neighbors in feature space.

| Parameter | Value | Reason |
|-----------|-------|--------|
| k | 7 (best from GridSearchCV over 3, 5, 7, 9, 11) | Optimal neighborhood size |
| Weighting | distance | Closer neighbors weigh more |
| Metric | Manhattan | Better than Euclidean for this feature space |

k-NN was tuned using **GridSearchCV with 5-fold stratified cross-validation**, optimizing for macro F1-score across 30 parameter combinations (k × weighting × metric).

---

### 4. Rejection Mechanism (Unknown Class)

Both classifiers support `predict_proba()`, which outputs a confidence score per class. If the highest confidence score is **below 0.5**, the input is classified as **Unknown** instead of forcing a prediction. This handles blurred frames, mixed materials, or objects outside the six trained classes.

The threshold is configurable via `CONFIDENCE_THRESHOLD` in `realtime_app.py`.

---

## Results

| Model | Best Configuration | Validation Accuracy |
|-------|-------------------|-------------------|
| SVM | C=5, RBF kernel, gamma=scale | 78.50% |
| k-NN | k=7, manhattan, distance weighting | 66.50% |

SVM was selected as the deployment model based on superior accuracy and faster prediction time.

### SVM Confusion Matrix
![SVM Confusion Matrix](models/svm_confusion_matrix.png)

### k-NN Confusion Matrix
![k-NN Confusion Matrix](models/knn_confusion_matrix.png)

### k-NN GridSearchCV Results
![k-NN CV Results](models/knn_cv_results.png)

---

## Evaluation

Models were evaluated on a held-out validation set of **600 images** (100 per class) that were never seen during training. The following metrics were computed:

- **Overall accuracy** — percentage of correctly classified images
- **Per-class precision** — of all predictions for a class, how many were correct
- **Per-class recall** — of all true instances of a class, how many were correctly identified
- **F1-score** — harmonic mean of precision and recall per class
- **Confusion matrix** — full breakdown of predictions vs true labels

### Per-Class F1 Comparison

| Class | SVM F1 | k-NN F1 |
|-------|--------|---------|
| Glass | 0.76 | 0.59 |
| Paper | 0.85 | 0.73 |
| Cardboard | 0.86 | 0.79 |
| Plastic | 0.81 | 0.67 |
| Metal | 0.73 | 0.60 |
| Trash | 0.72 | 0.62 |

---

## Key Findings

- SVM consistently outperforms k-NN across all six classes — the largest gap is in **cardboard** (0.86 vs 0.79) and **plastic** (0.81 vs 0.67)
- **Cardboard** became the easiest class after switching to 32×32 HOG cells, which captures its coarse structural pattern more effectively
- **Glass** remains the hardest class for both models due to its reflective surface and low saturation overlapping with other materials
- **Manhattan distance** significantly outperforms Euclidean for k-NN on this feature space, as it is less sensitive to outlier dimensions
- Switching HOG pixels-per-cell from **16×16 to 32×32** was the single biggest accuracy improvement — coarser cells capture more global shape structure which better distinguishes material categories
- Deep feature extraction (e.g. pretrained CNN features) would likely push accuracy further, as handcrafted descriptors have a natural ceiling on visually ambiguous classes

---

## Sample Output

Running `feature_extraction.py`:
```
processing training split...
  glass (label 0): 400 images
  paper (label 1): 400 images
  cardboard (label 2): 400 images
  plastic (label 3): 400 images
  metal (label 4): 400 images
  trash (label 5): 400 images
  X_train: (2400, 738) | y_train: (2400,)
processing validation split...
  X_val: (600, 738) | y_val: (600,)
fitting standard scaler on training data...
  scaling complete
  feature dim: 738
done. feature extraction complete.
```

Running `train_svm.py`:
```
loading feature vectors...
  X_train: (2400, 738) | X_val: (600, 738)
searching over c values (rbf kernel, gamma=scale)...

  c=1     → val accuracy = 76.33%
  c=5     → val accuracy = 78.50%
  c=10    → val accuracy = 78.33%
  ...

  best c: 5
  best val accuracy: 78.50%

per-class report:
              precision    recall  f1-score   support
       glass       0.77      0.74      0.76       100
       paper       0.86      0.83      0.85       100
   cardboard       0.82      0.90      0.86       100
     plastic       0.85      0.77      0.81       100
       metal       0.72      0.73      0.73       100
       trash       0.70      0.74      0.72       100
    accuracy                           0.79       600
```

Running `train_knn.py`:
```
loading feature vectors...
  X_train: (2400, 738) | y_train: (2400,)
  X_val:   (600, 738)  | y_val:   (600,)

running gridsearchcv...
  param grid: {'n_neighbors': [3,5,7,9,11], 'weights': ['uniform','distance'], 'metric': ['euclidean','manhattan','minkowski']}
  cv: stratifiedkfold(n_splits=5)
  scoring: f1_macro

  best params : {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
  best cv f1_macro: 0.6765

  validation accuracy : 66.50%
  validation f1_macro : 0.6680
```

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
│   ├── svm_model.pkl               # trained SVM model
│   ├── svm_confusion_matrix.png    # SVM confusion matrix
│   ├── knn_confusion_matrix.png    # k-NN confusion matrix
│   ├── knn_cv_results.png          # k-NN gridsearchcv results plot
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
| scikit-image | HOG, LBP, and GLCM feature extraction |
| NumPy | Feature vector storage and manipulation |
| joblib | Model serialization |
| Pillow | Image loading and augmentation |
| seaborn | Confusion matrix visualization |
| tqdm | Progress bars during training and extraction |

---

## Contributors

<table>
  <tr>
    <td align="center">
      <b>Haneen Hisham</b><br/>
      <a href="https://github.com/haneens5905">@haneens5905</a>
    </td>
    <td align="center">
      <b>Shaza Moatasem</b><br/>
      <a href="https://github.com/shaza-22">@shaza-22</a>
    </td>
    <td align="center">
      <b>Ziad Tarek</b><br/>
      <a href="https://github.com/ziad-91">@ziad-91</a>
    </td>
    <td align="center">
      <b>Seif Waleed</b><br/>
      <a href="https://github.com/Malware404seif">@Malware404seif</a>
    </td>
    <td align="center">
      <b>Mohamed Ahmed</b><br/>
      <a href="https://github.com/mohamed-hamza20">@mohamed-hamza20</a>
    </td>
  </tr>
</table>
