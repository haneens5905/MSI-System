# trains and evaluates a k-nn classifier on the extracted feature vectors
# run: python src/train_knn.py

import os
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe on servers
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -- configuration --
# paths are relative to this script so it works on any machine
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

CLASS_NAMES  = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# candidate hyperparameters to search over
K_VALUES     = [1, 3, 5, 7, 9, 11, 15]
WEIGHT_MODES = ["uniform", "distance"]


# -- data loading --

def load_features():
    # loads pre-scaled feature matrices saved by feature_extraction.py
    # raises an error if any required file is missing
    print("loading feature vectors...")

    required = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"]
    for fname in required:
        fpath = os.path.join(FEATURES_DIR, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"missing file: {fpath}\n"
                "run feature_extraction.py first."
            )

    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))

    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    return X_train, y_train, X_val, y_val


# -- hyperparameter search --

def run_experiment(X_train, y_train, X_val, y_val):
    # trains a k-nn for every combination of k and weighting scheme
    # returns results sorted by validation accuracy (best first)
    #
    # weighting schemes:
    #   uniform  — all k neighbors vote equally
    #   distance — closer neighbors weigh more (1/distance), better for noisy boundaries
    print("\nsearching over k values and weighting schemes...")
    print(f"  k values: {K_VALUES}")
    print(f"  weights:  {WEIGHT_MODES}\n")

    results = []

    for weight in WEIGHT_MODES:
        for k in tqdm(K_VALUES, desc=f"weight={weight}", leave=False, unit="k"):
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=weight,
                metric="euclidean",
                n_jobs=-1,  # parallelize distance computation across all cpu cores
            )
            knn.fit(X_train, y_train)

            # evaluate on validation set (no rejection — fair comparison across configs)
            y_pred_val = knn.predict(X_val)
            val_acc    = accuracy_score(y_val, y_pred_val)

            # also track training accuracy to detect overfitting
            # note: k=1 will always give train_acc=1.0, this is expected
            y_pred_tr = knn.predict(X_train)
            train_acc = accuracy_score(y_train, y_pred_tr)

            results.append({
                "k"         : k,
                "weight"    : weight,
                "train_acc" : round(train_acc, 4),
                "val_acc"   : round(val_acc,   4),
                "model"     : knn,
            })

            print(f"  k={k:<3} weight={weight:<10} train={train_acc:.4f}  val={val_acc:.4f}")

    # sort by validation accuracy so results[0] is always the best
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    return results


# -- evaluation helpers --

def print_classification_report(y_true, y_pred, title=""):
    # prints per-class precision, recall, and f1 for all 6 classes
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    if title:
        print(f"\n{title}")
    print(report)
    return report


def save_confusion_matrix(y_true, y_pred, save_path):
    # plots and saves a row-normalized confusion matrix
    # cells show raw counts, color intensity reflects per-class recall
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # annotate each cell with the raw count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted label", fontsize=12)
    ax.set_ylabel("true label", fontsize=12)
    ax.set_title("k-nn confusion matrix (row-normalized)", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


def save_experiment_plot(results, save_path):
    # line plot of validation accuracy vs k for each weighting scheme
    # the best overall configuration is highlighted with a gold star
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
        ax.plot(ks, accs, color=colors[weight], marker=markers[weight],
                linewidth=2, markersize=7, label=f"weight={weight}")

    # highlight the best configuration
    best = results[0]
    ax.scatter(
        [best["k"]], [best["val_acc"]],
        color="gold", edgecolors="black",
        s=200, zorder=5, linewidths=1.5,
        label=f"best: k={best['k']}, {best['weight']} ({best['val_acc']:.4f})",
    )

    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("validation accuracy", fontsize=12)
    ax.set_title("k-nn hyperparameter sweep", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


# -- main pipeline --

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # step 1: load pre-scaled feature vectors
    X_train, y_train, X_val, y_val = load_features()

    # step 2: search over all k and weighting combinations
    results = run_experiment(X_train, y_train, X_val, y_val)

    # step 3: print top 5 configurations
    print("\ntop 5 configurations:")
    print(f"  {'rank':<5} {'k':>3}  {'weight':<12} {'train_acc':>9} {'val_acc':>8}")
    for rank, r in enumerate(results[:5], start=1):
        print(f"  {rank:<5} {r['k']:>3}  {r['weight']:<12} {r['train_acc']:>9.4f} {r['val_acc']:>8.4f}")

    # step 4: evaluate best model on validation set
    best     = results[0]
    best_knn = best["model"]
    print(f"\nbest configuration: k={best['k']}, weight={best['weight']}, val accuracy={best['val_acc']:.4f}")

    y_pred_val = best_knn.predict(X_val)
    val_acc    = accuracy_score(y_val, y_pred_val)

    print(f"\nper-class report:")
    report_str = print_classification_report(
        y_val, y_pred_val,
        title=f"k-nn report [k={best['k']}, weight={best['weight']}]"
    )

    # save classification report as a text file
    report_path = os.path.join(MODELS_DIR, "knn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"k-nn classifier — material stream identification\n")
        f.write(f"best k={best['k']} | weight={best['weight']}\n\n")
        f.write(report_str)
        f.write(f"\noverall validation accuracy: {val_acc:.4f}\n")
    print(f"  saved: knn_classification_report.txt")

    # step 5: save confusion matrix and experiment plot
    save_confusion_matrix(y_val, y_pred_val, os.path.join(MODELS_DIR, "knn_confusion_matrix.png"))
    save_experiment_plot(results, os.path.join(MODELS_DIR, "knn_experiment_results.png"))

    # step 6: save the best model to disk
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    joblib.dump(best_knn, model_path)
    print(f"  saved: knn_model.pkl")

    print(f"\ndone. best k={best['k']}, weight={best['weight']}, final accuracy={val_acc:.4f}")
    return best_knn


if __name__ == "__main__":
    main()