"""
train_svm.py  —  Member 3
====================================
Trains an SVM classifier on the extracted feature vectors.

Run
---
    python src/train_svm.py
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm    import SVC
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix)

# ──────────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
CLASS_NAMES  = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]


def main():
    print("=" * 60)
    print("  SVM CLASSIFIER TRAINING  —  Member 3")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────────
    print("\n[1/4] Loading feature vectors ...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    print(f"       X_train : {X_train.shape}")
    print(f"       X_val   : {X_val.shape}")
    # Features already scaled by Member 2 — do NOT rescale

    # ── 2. Try a focused set of C values with RBF ──────────────────
    # From our experiments: kernel=rbf, gamma=scale is best.
    # We now do a manual search over C only — much faster and cleaner.
    print("\n[2/4] Manual search over C values (rbf, gamma=scale) ...")
    print("       Testing C = 1, 5, 10, 50, 100, 200, 500 ...\n")

    best_acc   = 0
    best_C     = None
    best_model = None

    for C in [1, 5, 10, 50, 100, 200, 500]:
        model = SVC(
            kernel="rbf",
            C=C,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"       C={C:>5}  →  val accuracy = {acc*100:.2f}%")

        if acc > best_acc:
            best_acc   = acc
            best_C     = C
            best_model = model

    print(f"\n       Best C     : {best_C}")
    print(f"       Best val acc : {best_acc*100:.2f}%")

    # ── 3. Evaluate best model ─────────────────────────────────────
    print("\n[3/4] Evaluating best model on validation set ...")
    y_pred = best_model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)

    print(f"\n       Validation Accuracy : {acc*100:.2f}%")
    print("\n-- Per-Class Report --")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_val, y_pred)
    print("-- Confusion Matrix --")
    print(cm)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"SVM Confusion Matrix — Validation Set  (acc={acc*100:.1f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(MODELS_DIR, "svm_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"\n       Plot saved: {plot_path}")
    plt.show()

    # ── 4. Save ────────────────────────────────────────────────────
    print("\n[4/4] Saving model ...")
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"       >> svm_model.pkl saved")

    print("\n" + "=" * 60)
    print(f"  BEST C          : {best_C}")
    print(f"  FINAL ACCURACY  : {acc*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
