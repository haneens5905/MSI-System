"""
knn_classifier.py  --  Member 4
====================================
Trains and evaluates a k-Nearest Neighbours (k-NN) classifier on the
feature vectors produced by feature_extraction
Experiments
-----------
  * k values       : 1, 3, 5, 7, 9, 11, 15
  * Weighting      : 'uniform'  and  'distance'
  -> Best combination is selected by validation accuracy.

Output
------
  models/
    knn_model.pkl          - best fitted KNeighborsClassifier

  reports/
    knn_classification_report.txt  - per-class precision/recall/F1
    knn_confusion_matrix.png       - heatmap (6 known classes)
    knn_experiment_results.png     - accuracy vs k plot
Run
---
    python src/knn_classifier.py
"""

import os
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on servers)
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics  import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# PATHS  (all relative to the project root)

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)          # src/ -> project root

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

# CONSTANTS

CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]


# Candidate hyper-parameters
K_VALUES     = [1, 3, 5, 7, 9, 11, 15]
WEIGHT_MODES = ["uniform", "distance"]



#  LOADING


def load_features():
    """
    Load the pre-computed, pre-scaled feature matrices saved by
    feature_extraction.py (Member 2).

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
    """
    print("[load] Reading feature files from:", FEATURES_DIR)

    required = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"]
    for fname in required:
        fpath = os.path.join(FEATURES_DIR, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Expected feature file not found: {fpath}\n"
                "Run feature_extraction.py (Member 2) first."
            )

    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))

    print(f"       X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"       X_val   : {X_val.shape}  |  y_val   : {y_val.shape}")
    print(f"       Feature dimension (D) = {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val



#  EXPERIMENT SWEEP  (k x weighting scheme)


def run_experiment(X_train, y_train, X_val, y_val):
    """
    Train a k-NN for every (k, weight) combination and evaluate on the
    validation set

    Weighting schemes
    -----------------
    * uniform  - all k neighbours contribute equally to the vote.
    * distance - closer neighbours weigh more (weight = 1/distance).
                 Better handles noisy or uneven class boundaries.

    Returns
    -------
    results : list[dict]  sorted descending by val_accuracy.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT SWEEP  --  k x weighting scheme")
    print("=" * 60)
    print(f"  k values     : {K_VALUES}")
    print(f"  weight modes : {WEIGHT_MODES}")
    print(f"  Total runs   : {len(K_VALUES) * len(WEIGHT_MODES)}")
    print("-" * 60)

    results = []

    for weight in WEIGHT_MODES:
        for k in tqdm(K_VALUES, desc=f"weight={weight}", leave=False, unit="k"):
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=weight,
                metric="euclidean",
                n_jobs=-1,           # parallelise distance computation
            )
            knn.fit(X_train, y_train)

            # Validation accuracy (no rejection -- fair comparison across configs)
            y_pred_val = knn.predict(X_val)
            val_acc    = accuracy_score(y_val, y_pred_val)

            # Training accuracy (k=1 will be 1.0 -- expected; watch the gap)
            y_pred_tr  = knn.predict(X_train)
            train_acc  = accuracy_score(y_train, y_pred_tr)

            results.append({
                "k"         : k,
                "weight"    : weight,
                "train_acc" : round(train_acc, 4),
                "val_acc"   : round(val_acc,   4),
                "model"     : knn,
            })

            print(
                f"  k={k:>2d}  weight={weight:<10s}"
                f"  train={train_acc:.4f}  val={val_acc:.4f}"
            )

    results.sort(key=lambda r: r["val_acc"], reverse=True)
    return results



#  EVALUATION HELPERS

def print_classification_report(y_true, y_pred, title=""):
    """Print a per-class precision / recall / F1 report for the 6 classes."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    if title:
        print(f"\n{'-'*60}\n{title}\n{'-'*60}")
    print(report)
    return report


def save_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot and save a normalised confusion matrix for the 6 known classes.

    Cells show raw counts; colour intensity reflects the row-normalised
    fraction (diagonal = per-class recall).
    """
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title("k-NN Confusion Matrix (row-normalised)", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] Confusion matrix -> {save_path}")


def save_experiment_plot(results, save_path):
    """
    Line plot of validation accuracy vs k for each weighting scheme.
    The best overall (k, weight) is highlighted with a gold star.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = {"uniform": "#2196F3", "distance": "#FF5722"}
    markers = {"uniform": "o",       "distance": "s"}

    for weight in WEIGHT_MODES:
        subset = sorted(
            [r for r in results if r["weight"] == weight],
            key=lambda r: r["k"],
        )
        ks   = [r["k"]       for r in subset]
        accs = [r["val_acc"] for r in subset]
        ax.plot(
            ks, accs,
            color=colors[weight], marker=markers[weight],
            linewidth=2, markersize=7, label=f"weight={weight}",
        )

    best = results[0]
    ax.scatter(
        [best["k"]], [best["val_acc"]],
        color="gold", edgecolors="black",
        s=200, zorder=5, linewidths=1.5,
        label=(
            f"Best: k={best['k']}, {best['weight']}"
            f"  ({best['val_acc']:.4f})"
        ),
    )

    ax.set_xlabel("k (number of neighbours)", fontsize=12)
    ax.set_ylabel("Validation Accuracy",       fontsize=12)
    ax.set_title("k-NN Hyper-parameter Sweep", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] Experiment plot  -> {save_path}")



#  MAIN


def main():
    print("=" * 60)
    print("  k-NN CLASSIFIER  --  Member 4")
    print("=" * 60)

    # -- 0. Setup output directories --------------------------------
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # -- 1. Load pre-computed features ------------------------------
    X_train, y_train, X_val, y_val = load_features()

    # -- 2. Hyper-parameter sweep ------------------------------------
    results = run_experiment(X_train, y_train, X_val, y_val)

    # -- 3. Top-5 summary -----------------------------------------------
    print("\n-- Top-5 Configurations ---------------------------------")
    print(f"  {'Rank':<5} {'k':>3}  {'weight':<12} {'train_acc':>9} {'val_acc':>8}")
    print(f"  {'-'*5} {'-'*3}  {'-'*12} {'-'*9} {'-'*8}")
    for rank, r in enumerate(results[:5], start=1):
        print(
            f"  {rank:<5} {r['k']:>3}  {r['weight']:<12}"
            f" {r['train_acc']:>9.4f} {r['val_acc']:>8.4f}"
        )

    # -- 5. Best model -----------------------------------------------
    best     = results[0]
    best_knn = best["model"]
    print(f"\n-- Best configuration ------------------------------------")
    print(f"   k      = {best['k']}")
    print(f"   weight = {best['weight']}")
    print(f"   val accuracy = {best['val_acc']:.4f}")

    # -- 6. Standard evaluation (no rejection) -----------------------
    #       Here we evaluate the raw classifier on the 6 known classes.
    y_pred_val = best_knn.predict(X_val)
    val_acc    = accuracy_score(y_val, y_pred_val)
    print(f"\n-- Validation Results (no rejection) ---------------------")
    print(f"   Overall accuracy : {val_acc:.4f}")

    # -- 7. Per-class classification report -------------------------
    report_str = print_classification_report(
        y_val, y_pred_val,
        title=(
            f"k-NN Classification Report  "
            f"[k={best['k']}, weight={best['weight']}]"
        ),
    )

    report_path = os.path.join(REPORTS_DIR, "knn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("k-NN Classifier -- Material Stream Identification\n")
        f.write(f"Best k = {best['k']}  |  weight = {best['weight']}\n\n")
        f.write(report_str)
        f.write(f"\nOverall validation accuracy : {val_acc:.4f}\n")
    print(f"[saved] Classification report ")

    # -- 8. Confusion matrix ------------------------------------------
    cm_path = os.path.join(REPORTS_DIR, "knn_confusion_matrix.png")
    save_confusion_matrix(y_val, y_pred_val, cm_path)

    # -- 9. Experiment plot -------------------------------------------
    plot_path = os.path.join(REPORTS_DIR, "knn_experiment_results.png")
    save_experiment_plot(results, plot_path)

    # -- 10. Save best model ---------------------------------------------
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    joblib.dump(best_knn, model_path)
    print(f"[saved] Best k-NN model  ")

    print("\n  k-NN training complete.")
    return best_knn


# =====================================================================
if __name__ == "__main__":
    main()

