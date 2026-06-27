"""
utils/dataset_utils.py
Dataset loading, preprocessing, and augmentation utilities for
AI-Based Edge-Enabled Farm Animal Intrusion Prevention System
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import urllib.request

# ─────────────────────────────────────────────
# ANIMAL CLASSES (customize to your dataset)
# ─────────────────────────────────────────────
ANIMAL_CLASSES = [
    "cow", "human", "goat", "deer", "wild_boar"
]
NUM_CLASSES = len(ANIMAL_CLASSES)
IMG_SIZE    = (224, 224)   # Standard input for MobileNet / Custom CNN


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (for demo/testing)
# Replace with real dataset loading in production
# ─────────────────────────────────────────────
def generate_synthetic_dataset(num_samples=500, img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Generates synthetic farm-like images for testing the pipeline.
    Uses evenly-spaced HSV hues so every class gets a distinct colour,
    scaling automatically to any number of classes.
    In a real project, replace with actual image loading from disk/Roboflow.
    """
    np.random.seed(42)
    X, y = [], []

    def _class_color(cls_idx, total):
        """Return a unique RGB colour per class using HSV hue spacing."""
        hue = int(cls_idx * 180 / total)          # OpenCV hue is 0-179
        hsv = np.uint8([[[hue, 200, 180]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # RGB

    for i in range(num_samples):
        cls = i % num_classes
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        base_color = _class_color(cls, num_classes)
        img[:] = base_color
        # Add noise to differentiate samples within a class
        noise = np.random.randint(-40, 40, img.shape, dtype=np.int16)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        # Draw a simple oval body shape
        cv2.ellipse(img,
                    (img_size[1]//2, img_size[0]//2),
                    (img_size[1]//4, img_size[0]//3),
                    0, 0, 360,
                    tuple(int(c * 0.7) for c in base_color), -1)
        X.append(img)
        y.append(cls)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# REAL DATASET LOADER (from directory)
# ─────────────────────────────────────────────
def load_dataset_from_directory(data_dir, img_size=IMG_SIZE):
    """
    Load images from a directory structure:
        data_dir/
            cow/   img1.jpg  img2.jpg ...
            dog/   img1.jpg  ...
            ...
    Returns X (float32 arrays), y (int labels), label_names
    """
    X, y, label_names = [], [], []
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    label_map = {cls: idx for idx, cls in enumerate(class_dirs)}
    label_names = class_dirs

    for cls_name in class_dirs:
        cls_path = os.path.join(data_dir, cls_name)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                fpath = os.path.join(cls_path, fname)
                try:
                    img = cv2.imread(fpath)
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(label_map[cls_name])
                except Exception as e:
                    print(f"[WARN] Could not load {fpath}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_names


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(X, normalize=True):
    X = X.astype(np.float32)
    if normalize:
        X = X / 255.0
    return X


def get_splits(X, y, test_size=0.2, val_size=0.1, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size),
        random_state=seed, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# TF DATA PIPELINE WITH AUGMENTATION
# ─────────────────────────────────────────────
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_tf_dataset(X, y, batch_size=32, augment_data=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
