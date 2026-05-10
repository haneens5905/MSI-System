# augments the raw dataset and balances all classes to 500 images each
# run: python src/augmentation.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# -- configuration --
# paths are relative to this script so it works on any machine
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

ORIGINAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "data/raw")
AUGMENTED_DATASET_PATH = os.path.join(PROJECT_ROOT, "data/augmented")
TARGET_IMAGES_PER_CLASS = 500
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


# -- helper functions --
def apply_random_augmentations(img):
    # randomly applies a mix of transforms to create a new image variant
    aug_img = img.copy()

    # random 90 degree rotations
    if np.random.rand() < 0.5:
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3 maps to 0, 90, 180, 270 degrees
        if k == 1:
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
        elif k == 2:
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)
        elif k == 3:
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # horizontal flip
    if np.random.rand() < 0.5:
        aug_img = cv2.flip(aug_img, 1)

    # vertical flip (less frequent than horizontal)
    if np.random.rand() < 0.3:
        aug_img = cv2.flip(aug_img, 0)

    # brightness and contrast adjustment
    if np.random.rand() < 0.7:
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        aug_img = cv2.convertScaleAbs(aug_img, alpha=contrast, beta=brightness * 50)

    # add gaussian noise to simulate real-world variation
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, np.random.uniform(10, 50), aug_img.shape)
        aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)

    # affine transformation (slight rotation, scale, and shift combined)
    if np.random.rand() < 0.6:
        h, w = aug_img.shape[:2]
        center = (w // 2, h // 2)
        angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.8, 1.2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # add a small random translation
        matrix[0, 2] += np.random.uniform(-w * 0.1, w * 0.1)
        matrix[1, 2] += np.random.uniform(-h * 0.1, h * 0.1)

        aug_img = cv2.warpAffine(aug_img, matrix, (w, h))

    return aug_img


def load_images_from_folder(folder_path):
    # loads all valid images from a folder into a list
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
    return images


def save_image(img, save_path):
    # saves a single image to disk
    cv2.imwrite(save_path, img)


def augment_images(images, target_count):
    # keeps all originals then generates new variants until we hit the target
    augmented = []
    original_count = len(images)

    # start with all original images
    augmented.extend(images)

    # if we already have enough originals, just return a subset
    if original_count >= target_count:
        return augmented[:target_count]

    # generate the remaining needed images
    needed = target_count - original_count
    current = original_count

    pbar = tqdm(total=needed, desc=f"augmenting {needed} images")
    while current < target_count:
        # pick a random original image to augment
        idx = np.random.randint(0, original_count)
        img = images[idx]

        # apply random augmentation and add to the list
        aug_img = apply_random_augmentations(img)
        augmented.append(aug_img)
        current += 1
        pbar.update(1)
    pbar.close()

    return augmented


def create_class_folders(base_path, class_names):
    # creates train/val subfolders for each class
    splits = ['train', 'val']
    for split in splits:
        for class_name in class_names:
            os.makedirs(os.path.join(base_path, split, class_name), exist_ok=True)


def save_split_images(split_path, class_name, images):
    # saves a list of images into the correct class subfolder
    for i, img in enumerate(images):
        save_path = os.path.join(split_path, class_name, f"{class_name}_{i}.jpg")
        save_image(img, save_path)


# -- main pipeline --
def main():
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    np.random.seed(RANDOM_SEED)  # fix seed so results are reproducible

    # wipe and recreate the augmented folder to start fresh
    if os.path.exists(AUGMENTED_DATASET_PATH):
        shutil.rmtree(AUGMENTED_DATASET_PATH)
    os.makedirs(AUGMENTED_DATASET_PATH)

    # create train/val subfolders for each class
    create_class_folders(AUGMENTED_DATASET_PATH, class_names)

    all_class_images = {}

    # step 1: load original images for each class
    print("loading original images...")
    for class_name in class_names:
        class_path = os.path.join(ORIGINAL_DATASET_PATH, class_name)
        images = load_images_from_folder(class_path)
        print(f"  {class_name}: {len(images)} images")
        all_class_images[class_name] = images

    # step 2: augment each class until we reach the target count
    print(f"\naugmenting to {TARGET_IMAGES_PER_CLASS} images per class...")
    augmented_class_images = {}
    for class_name in class_names:
        orig_imgs = all_class_images[class_name]
        augmented = augment_images(orig_imgs, TARGET_IMAGES_PER_CLASS)
        augmented_class_images[class_name] = augmented
        print(f"  {class_name}: {len(augmented)} images after augmentation")

    # step 3: split into train/val (80/20) and save to disk
    print("\nsplitting into train and val sets...")
    for class_name in class_names:
        images = augmented_class_images[class_name]
        train_imgs, val_imgs = train_test_split(images, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)

        # save each split to its folder
        train_path = os.path.join(AUGMENTED_DATASET_PATH, 'train')
        val_path = os.path.join(AUGMENTED_DATASET_PATH, 'val')

        save_split_images(train_path, class_name, train_imgs)
        save_split_images(val_path, class_name, val_imgs)

        print(f"  {class_name}: train={len(train_imgs)}, val={len(val_imgs)}")

    print(f"\ndone. augmented dataset saved to: {AUGMENTED_DATASET_PATH}")


if __name__ == "__main__":
    main()

