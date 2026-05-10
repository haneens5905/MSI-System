# trains and evaluates an svm classifier on the extracted feature vectors
# run: python src/train_svm.py

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix)

# -- configuration --
# paths are relative to this script so it works on any machine
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
CLASS_NAMES  = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]


def main():

    # step 1: load pre-scaled feature vectors saved by feature_extraction.py
    print("loading feature vectors...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape}")
    # features are already scaled by feature_extraction.py — do not rescale

    # step 2: search over a range of c values to find the best regularization strength
    # rbf kernel with gamma=scale was chosen based on experimentation
    # c controls the trade-off between a smooth decision boundary and classifying training points correctly
    print("\nsearching over c values (rbf kernel, gamma=scale)...")
    print("testing c = 1, 5, 10, 50, 100, 200, 500...\n")

    best_acc   = 0
    best_C     = None
    best_model = None

    for C in [1, 5, 10, 50, 100, 200, 500]:
        model = SVC(
            kernel="rbf",       # rbf handles non-linear boundaries between material classes
            C=C,
            gamma="scale",      # automatically scales gamma based on feature variance
            class_weight="balanced",  # compensates for any remaining class imbalance
            probability=True,   # needed for the confidence-based rejection mechanism
            random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  c={C:<5} → val accuracy = {acc*100:.2f}%")

        # keep track of the best performing model
        if acc > best_acc:
            best_acc   = acc
            best_C     = C
            best_model = model

    print(f"\n  best c: {best_C}")
    print(f"  best val accuracy: {best_acc*100:.2f}%")

    # step 3: evaluate the best model on the validation set
    print("\nevaluating best model...")
    y_pred = best_model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)

    print(f"\n  validation accuracy: {acc*100:.2f}%")
    print("\nper-class report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

    # print and plot the confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("confusion matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"svm confusion matrix — validation set (acc={acc*100:.1f}%)")
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(MODELS_DIR, "svm_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"\n  confusion matrix saved to: {plot_path}")
    plt.show()

    # step 4: save the best model to disk
    print("\nsaving model...")
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  saved: svm_model.pkl")
    print(f"\ndone. best c={best_C}, final accuracy={acc*100:.2f}%")


if __name__ == "__main__":
    main()