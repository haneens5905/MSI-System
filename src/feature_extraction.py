"""
feature_extraction.py  —  Member 2
====================================
Converts augmented images into fixed-length numerical feature vectors
using three complementary descriptors:
    1. HOG   (Histogram of Oriented Gradients)  → shape & edge structure
    2. HSV Color Histogram                      → colour distribution
    3. LBP   (Local Binary Patterns)            → micro-texture

Output
------
features/
    X_train.npy   – training feature matrix   (N_train × D)
    X_val.npy     – validation feature matrix  (N_val   × D)
    y_train.npy   – training labels            (N_train,)
    y_val.npy     – validation labels          (N_val,)
    scaler.pkl    – StandardScaler fitted on the training set

Run
---
    python src/feature_extraction.py
"""

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ──────────────────────────────────────────────
# PATHS  (all relative to the project root)
# ──────────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # src/ → project root

AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "augmented")
FEATURES_DIR  = os.path.join(PROJECT_ROOT, "features")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
IMG_SIZE = (128, 128)          # Fixed resize for every image (width, height)

# Class names must match the folder names produced by augmentation.py
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# Label mapping  (class 6 = Unknown is handled by classifier rejection)
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# --- HOG parameters ---
HOG_ORIENTATIONS  = 9
HOG_PIX_PER_CELL  = (16, 16)
HOG_CELLS_PER_BLK = (2, 2)

# --- HSV histogram parameters ---
HSV_BINS = [8, 8, 8]           # bins for H, S, V channels  → 512 total

# --- LBP parameters ---
LBP_RADIUS = 3
LBP_N_POINTS = 24              # 8 × radius is a common choice
LBP_METHOD = "uniform"         # reduces feature bins to (n_points + 2)


# =====================================================================
#  FEATURE DESCRIPTORS
# =====================================================================

def compute_hog(gray):
    """
    Histogram of Oriented Gradients.

    HOG divides the image into small cells and, for each cell, computes a
    histogram of gradient directions (edges).  Adjacent cells are grouped
    into overlapping blocks whose histograms are contrast-normalised.

    This descriptor captures **shape and structural information** — vital
    for distinguishing bottles (round edges), cans (straight edges),
    crumpled paper (irregular edges), and flat cardboard.

    Parameters used
    ---------------
    orientations : 9   – each bin covers 20° (0–180°)
    pixels_per_cell : (16, 16) – coarser cells keep the vector compact
    cells_per_block : (2, 2)   – 2×2 cells per block for normalisation

    For a 128×128 image → 8×8 cell grid → 7×7 block grid
    Feature length = 9 × 4 × 49 = 1764
    """
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIX_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLK,
        block_norm="L2-Hys",
        transform_sqrt=True,       # gamma correction for better contrast
        feature_vector=True,
    )
    return features


def compute_color_histogram(bgr):
    """
    3-D colour histogram in the HSV colour space.

    HSV separates **hue** (actual colour) from **saturation** and **value**
    (lighting), making the histogram more robust to illumination changes
    than an RGB histogram.

    Why it matters for material sorting
    ------------------------------------
    • Metal objects have a narrow, distinctive hue range (silver/grey).
    • Glass is often clear/green/brown with low saturation.
    • Paper tends toward white (low S, high V).
    • Plastic comes in many bright colours (high S).
    • Trash has mixed, dull colours.

    Bins : 8 × 8 × 8 = 512
    Normalised with L1-norm so that the histogram sums to 1.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, HSV_BINS,
        [0, 180, 0, 256, 0, 256],    # OpenCV HSV ranges
    )
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist


def compute_lbp(gray):
    """
    Local Binary Pattern histogram (uniform variant).

    For every pixel, LBP compares it with its circular neighbours and
    encodes the result as a binary number.  The "uniform" variant keeps
    only the patterns that have at most two 0→1 or 1→0 transitions,
    which correspond to meaningful micro-structures (edges, corners,
    flat areas) while discarding noisy patterns.

    Why it matters for material sorting
    ------------------------------------
    • Cardboard has a coarse, fibrous texture → many edge-type LBP codes.
    • Plastic bags / bottles are smooth → dominated by flat-area codes.
    • Metal cans have a fine-grained, reflective texture.
    • Glass surfaces produce specular highlights → distinctive LBP profile.

    Parameters
    ----------
    radius : 3       – captures texture at a slightly coarser scale
    n_points : 24    – 24 neighbours on the circle of radius 3
    method : uniform – yields (n_points + 2) = 26 histogram bins
    """
    lbp_img = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)

    # Build a normalised histogram of the LBP codes
    n_bins = int(LBP_N_POINTS + 2)   # uniform LBP has n_points + 2 bins
    hist, _ = np.histogram(
        lbp_img.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )
    return hist


# =====================================================================
#  COMBINED FEATURE VECTOR
# =====================================================================

def extract_features(bgr_image):
    """
    Convert a single BGR image into a fixed-length 1-D feature vector.

    Pipeline
    --------
    1. Resize to IMG_SIZE (128×128).
    2. Convert to grayscale (for HOG and LBP).
    3. Compute HOG         → 1764 dims
    4. Compute HSV histogram → 512 dims
    5. Compute LBP histogram →  26 dims
    6. Concatenate           → 2302 dims total

    Parameters
    ----------
    bgr_image : np.ndarray
        Image loaded by cv2.imread (BGR, uint8).

    Returns
    -------
    np.ndarray of shape (2302,) — dtype float64.
    """
    # 1. Resize
    resized = cv2.resize(bgr_image, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3–5. Descriptors
    hog_feat   = compute_hog(gray)
    color_feat = compute_color_histogram(resized)
    lbp_feat   = compute_lbp(gray)

    # 6. Concatenate
    feature_vector = np.concatenate([hog_feat, color_feat, lbp_feat])
    return feature_vector


def extract_single_image(image_path):
    """
    Convenience wrapper: load an image file and return its feature vector.

    Useful for the real-time app (Member 5) and quick sanity checks.
    Returns None if the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Could not load image: {image_path}")
        return None
    return extract_features(img)


# =====================================================================
#  BATCH PROCESSING
# =====================================================================

def process_split(split_dir, class_names):
    """
    Walk through a split folder (train/ or val/) and extract features
    for every image.

    Parameters
    ----------
    split_dir : str
        Path to the split directory (e.g. data/augmented/train/).
    class_names : list[str]
        Ordered list of class folder names.

    Returns
    -------
    X : np.ndarray of shape (N, D) — feature matrix.
    y : np.ndarray of shape (N,)   — integer labels.
    """
    features_list = []
    labels_list   = []

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Missing class folder: {class_dir}")
            continue

        label = LABEL_MAP[class_name]
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"  {class_name:>10s} (label {label}): {len(image_files)} images")

        for fname in tqdm(image_files, desc=f"    {class_name}", leave=False):
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            vec = extract_features(img)
            features_list.append(vec)
            labels_list.append(label)

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=np.int32)
    return X, y


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("  FEATURE EXTRACTION  —  Member 2")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # ── 1. Process training split ──────────────────────────────────
    train_dir = os.path.join(AUGMENTED_DIR, "train")
    print(f"\n[1/4] Processing TRAINING split: {train_dir}")
    X_train, y_train = process_split(train_dir, CLASS_NAMES)
    print(f"       X_train shape: {X_train.shape}  |  y_train shape: {y_train.shape}")

    # ── 2. Process validation split ────────────────────────────────
    val_dir = os.path.join(AUGMENTED_DIR, "val")
    print(f"\n[2/4] Processing VALIDATION split: {val_dir}")
    X_val, y_val = process_split(val_dir, CLASS_NAMES)
    print(f"       X_val   shape: {X_val.shape}  |  y_val   shape: {y_val.shape}")

    # ── 3. Feature scaling (StandardScaler) ────────────────────────
    print("\n[3/4] Fitting StandardScaler on training data ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    print("       Scaling complete.")

    # ── 4. Save everything ─────────────────────────────────────────
    print(f"\n[4/4] Saving to {FEATURES_DIR}/")

    np.save(os.path.join(FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(FEATURES_DIR, "y_val.npy"),   y_val)
    joblib.dump(scaler, os.path.join(FEATURES_DIR, "scaler.pkl"))

    print(f"       >> X_train.npy  {X_train.shape}")
    print(f"       >> X_val.npy    {X_val.shape}")
    print(f"       >> y_train.npy  {y_train.shape}")
    print(f"       >> y_val.npy    {y_val.shape}")
    print(f"       >> scaler.pkl")

    # ── Quick sanity checks ────────────────────────────────────────
    print("\n-- Sanity Checks --")
    print(f"   NaN in X_train : {np.isnan(X_train).any()}")
    print(f"   NaN in X_val   : {np.isnan(X_val).any()}")
    print(f"   Inf in X_train : {np.isinf(X_train).any()}")
    print(f"   Inf in X_val   : {np.isinf(X_val).any()}")
    print(f"   Train labels   : {np.unique(y_train)}")
    print(f"   Val   labels   : {np.unique(y_val)}")
    print(f"   Feature dim    : {X_train.shape[1]}")

    # Label distribution
    print("\n-- Label Distribution --")
    for name in CLASS_NAMES:
        lbl = LABEL_MAP[name]
        n_tr = np.sum(y_train == lbl)
        n_vl = np.sum(y_val == lbl)
        print(f"   {name:>10s}  train={n_tr:4d}  val={n_vl:4d}")

    print("\n  Feature extraction complete.\n")


if __name__ == "__main__":
    main()
