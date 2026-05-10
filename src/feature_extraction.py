# converts augmented images into fixed-length numerical feature vectors using hog, hsv histogram, and lbp
# run: python src/feature_extraction.py

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -- configuration --
# paths are relative to this script so it works on any machine
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "augmented")
FEATURES_DIR  = os.path.join(PROJECT_ROOT, "features")

# fixed size all images are resized to before feature extraction
IMG_SIZE = (128, 128)

# class names must match the folder names from augmentation.py
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# maps each class name to an integer label
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# -- hog parameters --
HOG_ORIENTATIONS  = 9
HOG_PIX_PER_CELL  = (16, 16)
HOG_CELLS_PER_BLK = (2, 2)

# -- hsv histogram parameters --
HSV_BINS = [8, 8, 8]  # bins for h, s, v channels → 512 total

# -- lbp parameters --
LBP_RADIUS   = 3
LBP_N_POINTS = 24      # 8 × radius is the standard choice
LBP_METHOD   = "uniform"


# -- feature descriptors --

def compute_hog(gray):
    # hog captures shape and edge structure by computing gradient direction histograms across the image
    # for a 128x128 image this gives a 1764-dim vector
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIX_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLK,
        block_norm="L2-Hys",
        transform_sqrt=True,   # gamma correction for better contrast handling
        feature_vector=True,
    )
    return features


def compute_color_histogram(bgr):
    # hsv color histogram captures color distribution while being robust to lighting changes
    # using hsv instead of rgb separates hue (actual color) from brightness
    # 8x8x8 bins give a 512-dim vector, normalized so it sums to 1
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, HSV_BINS,
        [0, 180, 0, 256, 0, 256],  # opencv hsv ranges
    )
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist


def compute_lbp(gray):
    # lbp captures surface texture by comparing each pixel to its circular neighbors
    # the uniform variant keeps only meaningful patterns (edges, corners, flat areas)
    # gives a 26-dim normalized histogram
    lbp_img = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)

    n_bins = int(LBP_N_POINTS + 2)  # uniform lbp always has n_points + 2 bins
    hist, _ = np.histogram(
        lbp_img.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )
    return hist


# -- combined feature vector --

def extract_features(bgr_image):
    # converts a single bgr image into a fixed-length 1d feature vector
    # pipeline: resize → grayscale → hog (1764) + hsv hist (512) + lbp (26) → concat → 2302 dims total

    # resize to fixed size
    resized = cv2.resize(bgr_image, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # convert to grayscale for hog and lbp
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # compute all three descriptors
    hog_feat   = compute_hog(gray)
    color_feat = compute_color_histogram(resized)
    lbp_feat   = compute_lbp(gray)

    # concatenate into one vector
    feature_vector = np.concatenate([hog_feat, color_feat, lbp_feat])
    return feature_vector


def extract_single_image(image_path):
    # convenience wrapper used by the real-time app to extract features from one image file
    # note: the scaler must be applied separately before passing to the classifier
    img = cv2.imread(image_path)
    if img is None:
        print(f"warning: could not load image: {image_path}")
        return None
    return extract_features(img)


# -- batch processing --

def process_split(split_dir, class_names):
    # walks through a split folder and extracts features for every image
    # returns feature matrix x of shape (n, 2302) and label array y of shape (n,)
    features_list = []
    labels_list   = []

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"warning: missing class folder: {class_dir}")
            continue

        label = LABEL_MAP[class_name]
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"  {class_name} (label {label}): {len(image_files)} images")

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


# -- main pipeline --

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # step 1: extract features from training images
    train_dir = os.path.join(AUGMENTED_DIR, "train")
    print("processing training split...")
    X_train, y_train = process_split(train_dir, CLASS_NAMES)
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")

    # step 2: extract features from validation images
    val_dir = os.path.join(AUGMENTED_DIR, "val")
    print("\nprocessing validation split...")
    X_val, y_val = process_split(val_dir, CLASS_NAMES)
    print(f"  X_val: {X_val.shape} | y_val: {y_val.shape}")

    # step 3: fit scaler on training data only then apply to both splits
    # important: scaler is never fitted on val data to avoid data leakage
    print("\nfitting standard scaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    print("  scaling complete")

    # step 4: save feature matrices, labels, and scaler to disk
    print(f"\nsaving to {FEATURES_DIR}/")
    np.save(os.path.join(FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(FEATURES_DIR, "y_val.npy"),   y_val)
    joblib.dump(scaler, os.path.join(FEATURES_DIR, "scaler.pkl"))
    print("  saved: X_train.npy, X_val.npy, y_train.npy, y_val.npy, scaler.pkl")

    # sanity checks to make sure nothing went wrong
    print("\nsanity checks:")
    print(f"  nan in X_train : {np.isnan(X_train).any()}")
    print(f"  nan in X_val   : {np.isnan(X_val).any()}")
    print(f"  inf in X_train : {np.isinf(X_train).any()}")
    print(f"  inf in X_val   : {np.isinf(X_val).any()}")
    print(f"  feature dim    : {X_train.shape[1]}")
    print(f"  train labels   : {np.unique(y_train)}")
    print(f"  val labels     : {np.unique(y_val)}")

    # label distribution across train and val
    print("\nlabel distribution:")
    for name in CLASS_NAMES:
        lbl = LABEL_MAP[name]
        n_tr = np.sum(y_train == lbl)
        n_vl = np.sum(y_val == lbl)
        print(f"  {name}: train={n_tr}, val={n_vl}")

    print("\ndone. feature extraction complete.")


if __name__ == "__main__":
    main()