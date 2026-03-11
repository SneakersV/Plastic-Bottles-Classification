import os
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import torch
import random


# ======================= Seed =======================

SEED = 42

def set_seed(seed=SEED):
    """
    Fix all random seeds for reproducibility.
    Covers: Python random, NumPy, PyTorch (CPU + CUDA), cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # For multi-GPU
    torch.backends.cudnn.deterministic = True   # Ensure deterministic algorithms
    torch.backends.cudnn.benchmark = False      # Disable auto-tuner for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"  [Seed fixed: {seed}]")


# ======================= CSV & Split =======================

def read_split_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def get_split_dataframes(df):
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


# =========== Image Loading for LogReg / SVM (flatten) ===========

def load_image(path, image_size=(128, 128), show=False):
    """Load a single image, resize, normalize to [0,1], and flatten to 1D."""
    img = Image.open(path).convert("RGB")
    img_resized = img.resize(image_size)
    img_normalize = np.asarray(img_resized, dtype=np.float32) / 255.0
    img_flatten = img_normalize.reshape(1, -1)

    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    return img_flatten


def load_dataset(df, image_size=(128, 128)):
    """Load all images from a DataFrame. Returns flattened X (N, H*W*3) and y (N,)."""
    X = []
    y = []

    for _, row in df.iterrows():
        img_path = row['filepath']
        label = int(row['label'])

        try:
            img_flatten = load_image(img_path, image_size=image_size, show=False)
            X.append(img_flatten)
            y.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    X = np.vstack(X)
    y = np.array(y)

    return X, y


# =========== Data Augmentation for LogReg / SVM ===========

def _pil_augment(img):
    """
    Generate augmented versions of a single PIL image.
    Returns a list of augmented PIL images.
    """
    augmented = []

    # 1. Horizontal Flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # 2. Random Rotation (between -15 and +15 degrees)
    angle = random.uniform(-15, 15)
    augmented.append(img.rotate(angle, fillcolor=(0, 0, 0)))

    # 3. Brightness adjustment (0.7x to 1.3x)
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(random.uniform(0.7, 1.3)))

    # 4. Contrast adjustment (0.7x to 1.3x)
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(random.uniform(0.7, 1.3)))

    return augmented


def load_dataset_augmented(df, image_size=(128, 128)):
    """
    Load images WITH augmentation for sklearn models (training only).
    For each original image, generates 4 augmented copies:
      - Horizontal flip
      - Rotation (±15°)
      - Brightness jitter
      - Contrast jitter
    Returns flattened X (N*5, H*W*3) and y (N*5,).
    """
    random.seed(42)
    X = []
    y = []

    for _, row in df.iterrows():
        img_path = row['filepath']
        label = int(row['label'])

        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize(image_size)

            # Original image
            img_norm = np.asarray(img_resized, dtype=np.float32) / 255.0
            X.append(img_norm.reshape(1, -1))
            y.append(label)

            # Augmented copies
            aug_images = _pil_augment(img_resized)
            for aug_img in aug_images:
                aug_norm = np.asarray(aug_img, dtype=np.float32) / 255.0
                X.append(aug_norm.reshape(1, -1))
                y.append(label)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    X = np.vstack(X)
    y = np.array(y)

    print(f"  Original: {len(df)} images -> Augmented: {len(X)} samples (5x)")
    return X, y


# =========== Image Loading for CNN (tensor format) ===========

def load_dataset_for_cnn(df, image_size=(128, 128)):
    """
    Load images as tensors for PyTorch CNN (without transforms).
    Returns X as tensor (N, 3, H, W) and y as tensor (N,).
    Note: For training with augmentation, use BottleDataset with transforms instead.
    """
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = row['filepath']
        label = int(row['label'])

        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize(image_size)
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
            # Convert from (H, W, C) to (C, H, W) for PyTorch
            img_tensor = np.transpose(img_array, (2, 0, 1))
            images.append(img_tensor)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    X = torch.tensor(np.array(images), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)

    return X, y