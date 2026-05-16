# converts augmented images into fixed-length numerical feature vectors
# uses hog, color histograms (bgr+hsv+lab), color moments, lbp, glcm, hu moments, edge features, and gabor
# run: python src/feature_extraction.py

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
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
HOG_PIX_PER_CELL  = (32, 32)
HOG_CELLS_PER_BLK = (2, 2)

# -- glcm parameters --
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS     = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# -- lbp parameters --
LBP_RADIUS   = 3
LBP_N_POINTS = 24
LBP_METHOD   = "uniform"


# -- feature descriptors --

def compute_hog(gray):
    # hog captures shape and edge structure → 1764 dims
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIX_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )


def compute_color_histograms(bgr):
    # computes 32-bin histograms for each channel in bgr, hsv, and lab color spaces
    # using multiple color spaces makes the descriptor more robust and discriminative → 288 dims
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    features = []
    for image in [bgr, hsv, lab]:
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [32], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)  # normalize to avoid scale differences
            features.extend(hist)

    return np.array(features)


def compute_color_moments(bgr):
    # computes mean, std, and median for each hsv channel
    # compact but effective summary of color distribution → 9 dims
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    features = []
    for channel in range(3):
        pixels = hsv[:, :, channel].astype(np.float32)
        features.append(np.mean(pixels))
        features.append(np.std(pixels))
        features.append(np.median(pixels))
    return np.array(features)


def compute_lbp(gray):
    # lbp captures surface texture by comparing each pixel to its circular neighbors → 26 dims
    lbp_img = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = int(LBP_N_POINTS + 2)
    hist, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-7)


def compute_glcm(gray):
    # glcm captures texture properties at 3 distances and 4 angles → 72 dims
    # using 6 properties instead of 4 gives a richer texture description
    if gray.max() > 0:
        gray_uint8 = (gray / gray.max() * 255).astype(np.uint8)
    else:
        gray_uint8 = gray.astype(np.uint8)

    glcm = graycomatrix(
        gray_uint8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)

    return np.array(features)


def compute_hu_moments(gray):
    # hu moments capture shape properties that are invariant to rotation, scale, and translation → 7 dims
    # log-transformed to compress the large range of hu moment values
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def compute_edge_features(gray):
    # uses canny edge detection to capture edge density and contour statistics → 4 dims
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.array([edge_density, 0, 0, 0])

    areas = [cv2.contourArea(c) for c in contours]
    return np.array([
        edge_density,
        len(contours),
        np.mean(areas),
        np.std(areas)
    ])


def compute_gabor(gray):
    # applies gabor filters at 4 orientations, returns mean and std of each response → 8 dims
    features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel(
            (21, 21),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.append(filtered.mean())
        features.append(filtered.std())
    return np.array(features)


# -- combined feature vector --

def extract_features(bgr_image):
    # converts a single bgr image into a fixed-length 1d feature vector
    # pipeline: resize → grayscale → all descriptors → concatenate → 2178 dims total
    # hog (1764) + color hist (288) + color moments (9) + lbp (26) + glcm (72) + hu (7) + edges (4) + gabor (8)

    # resize to fixed size
    resized = cv2.resize(bgr_image, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # convert to grayscale for hog, lbp, glcm, hu moments, edge features, and gabor
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # compute all descriptors
    hog_feat    = compute_hog(gray)
    color_hist  = compute_color_histograms(resized)
    color_mom   = compute_color_moments(resized)
    lbp_feat    = compute_lbp(gray)
    glcm_feat   = compute_glcm(gray)
    hu_feat     = compute_hu_moments(gray)
    edge_feat   = compute_edge_features(gray)
    gabor_feat  = compute_gabor(gray)

    # concatenate into one vector
    feature_vector = np.concatenate([
        hog_feat, color_hist, color_mom, lbp_feat,
        glcm_feat, hu_feat, edge_feat, gabor_feat
    ])
    return feature_vector.astype(np.float32)


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
    # returns feature matrix x of shape (n, 2178) and label array y of shape (n,)
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

    X = np.array(features_list, dtype=np.float32)
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

    # sanity checks
    print("\nsanity checks:")
    print(f"  nan in X_train : {np.isnan(X_train).any()}")
    print(f"  nan in X_val   : {np.isnan(X_val).any()}")
    print(f"  inf in X_train : {np.isinf(X_train).any()}")
    print(f"  inf in X_val   : {np.isinf(X_val).any()}")
    print(f"  feature dim    : {X_train.shape[1]}")

    # label distribution
    print("\nlabel distribution:")
    for name in CLASS_NAMES:
        lbl = LABEL_MAP[name]
        n_tr = np.sum(y_train == lbl)
        n_vl = np.sum(y_val == lbl)
        print(f"  {name}: train={n_tr}, val={n_vl}")

    print("\ndone. feature extraction complete.")


if __name__ == "__main__":
    main()