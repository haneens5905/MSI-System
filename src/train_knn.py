# trains and evaluates a k-nn classifier on the extracted feature vectors
# uses gridsearchcv with stratified 5-fold cross-validation to find the best hyperparameters
# scoring is based on f1_macro which penalises weak classes harder than plain accuracy
# run: python src/train_knn.py

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe on servers
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -- configuration --
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

CLASS_NAMES  = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# search space — metric is the key addition over the previous version
# manhattan distance handles high-dimensional feature vectors better than euclidean
# because it is less sensitive to outlier dimensions
PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan", "minkowski"],
}


# -- data loading --

def load_features():
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


# -- evaluation helpers --

def print_classification_report(y_true, y_pred, title=""):
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
    ax.set_xlabel("predicted label", fontsize=12)
    ax.set_ylabel("true label", fontsize=12)
    ax.set_title("k-nn confusion matrix (row-normalized)", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


def save_cv_results_plot(grid_search, save_path):
    # plots mean cv f1_macro for every (k, weight, metric) combination
    # gives a visual summary of how each setting performed across folds
    results  = grid_search.cv_results_
    params   = results["params"]
    scores   = results["mean_test_score"]

    metrics  = PARAM_GRID["metric"]
    weights  = PARAM_GRID["weights"]
    colors   = {"uniform": "#2196F3", "distance": "#FF5722"}
    markers  = {"euclidean": "o", "manhattan": "s", "minkowski": "^"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=True)

    for ax, metric in zip(axes, metrics):
        for weight in weights:
            subset = [
                (p["n_neighbors"], s)
                for p, s in zip(params, scores)
                if p["metric"] == metric and p["weights"] == weight
            ]
            subset.sort(key=lambda x: x[0])
            ks    = [x[0] for x in subset]
            accs  = [x[1] for x in subset]
            ax.plot(ks, accs,
                    color=colors[weight],
                    marker=markers[metric],
                    linewidth=2, markersize=7,
                    label=f"weight={weight}")

        ax.set_title(f"metric={metric}", fontsize=11)
        ax.set_xlabel("k (neighbors)", fontsize=10)
        ax.set_xticks(PARAM_GRID["n_neighbors"])
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("mean cv f1_macro", fontsize=10)
    axes[0].legend(fontsize=9)
    fig.suptitle("k-nn gridsearchcv results", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


# -- main pipeline --

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # step 1: load feature vectors
    X_train, y_train, X_val, y_val = load_features()

    # step 2: gridsearchcv with stratified 5-fold cross-validation
    # stratified keeps the same class ratio in every fold
    # f1_macro scoring pushes the model to improve on weak classes, not just easy ones
    # n_jobs=-1 parallelises distance computation across all cpu cores
    print(f"\nrunning gridsearchcv...")
    print(f"  param grid: {PARAM_GRID}")
    print(f"  cv: stratifiedkfold(n_splits=5)")
    print(f"  scoring: f1_macro\n")

    knn = KNeighborsClassifier(n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  best params : {grid_search.best_params_}")
    print(f"  best cv f1_macro: {grid_search.best_score_:.4f}")

    # step 3: evaluate best model on held-out validation set
    best_knn = grid_search.best_estimator_

    print("\nevaluating best model on validation set...")
    y_pred_val = best_knn.predict(X_val)

    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1  = f1_score(y_val, y_pred_val, average="macro")

    print(f"\n  validation accuracy : {val_acc*100:.2f}%")
    print(f"  validation f1_macro : {val_f1:.4f}")

    report_str = print_classification_report(
        y_val, y_pred_val,
        title=f"k-nn report [{grid_search.best_params_}]"
    )

    # save classification report as a text file
    report_path = os.path.join(MODELS_DIR, "knn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"k-nn classifier — material stream identification\n")
        f.write(f"best params: {grid_search.best_params_}\n\n")
        f.write(report_str)
        f.write(f"\noverall validation accuracy : {val_acc:.4f}\n")
        f.write(f"overall validation f1_macro : {val_f1:.4f}\n")
    print(f"  saved: knn_classification_report.txt")

    # step 4: save confusion matrix and cv results plot
    save_confusion_matrix(y_val, y_pred_val, os.path.join(MODELS_DIR, "knn_confusion_matrix.png"))
    save_cv_results_plot(grid_search, os.path.join(MODELS_DIR, "knn_cv_results.png"))

    # step 5: save best model
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    joblib.dump(best_knn, model_path)
    print(f"  saved: knn_model.pkl")

    print(f"\ndone. best params={grid_search.best_params_}, accuracy={val_acc*100:.2f}%, f1={val_f1:.4f}")
    return best_knn


if __name__ == "__main__":
    main()