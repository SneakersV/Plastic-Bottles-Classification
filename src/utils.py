import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


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


# =========== Image Loading for CNN (tensor format) ===========

def load_dataset_for_cnn(df, image_size=(128, 128)):
    """
    Load images as tensors for PyTorch CNN.
    Returns X as tensor (N, 3, H, W) and y as tensor (N,).
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