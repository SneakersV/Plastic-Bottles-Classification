import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import tomllib


# ======================= Config =======================

def load_config(config_path="config.toml"):
    """Load TOML configuration file."""
    # Try finding the config in current dir or project root
    if not os.path.exists(config_path):
        current_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        config_path = os.path.join(project_root, "config.toml")
        
    with open(config_path, "rb") as f:
        return tomllib.load(f)

# Global access to config
config = load_config()

# ======================= Seed =======================

SEED = config["training"]["random_seed"]

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


# ======================= Training Plots =======================

def plot_training_history(history, model_name, save_dir="plots"):
    """
    Plot training vs validation loss and F1 score curves.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_f1', 'val_f1'
                 (each is a list of values per epoch)
        model_name: str, name of the model (used for title and filename)
        save_dir: str, directory to save the plot image
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss plot ---
    axes[0].plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- F1 Score plot ---
    axes[1].plot(epochs, history['train_f1'], 'b-o', markersize=4, label='Train F1')
    axes[1].plot(epochs, history['val_f1'], 'r-o', markersize=4, label='Val F1')
    axes[1].set_title(f'{model_name} - F1 Score (weighted)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_training_history.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  📊 Training plot saved to: {save_path}")
    return save_path


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


# =========== Data Augmentation Functions ===========

def add_blur(img, radius=2):
    """Add Gaussian Blur to the image."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_vertical_strip(img, strip_width_ratio=0.15):
    """Add a black vertical strip to the image to simulate occlusion."""
    img_np = np.array(img).copy()
    H, W, C = img_np.shape
    strip_width = max(1, int(W * strip_width_ratio))
    x = random.randint(0, W - strip_width)
    img_np[:, x:x+strip_width, :] = 0
    return Image.fromarray(img_np)

def add_horizontal_strip(img, strip_height_ratio=0.15):
    """Add a black horizontal strip to the image to simulate occlusion."""
    img_np = np.array(img).copy()
    H, W, C = img_np.shape
    strip_height = max(1, int(H * strip_height_ratio))
    y = random.randint(0, H - strip_height)
    img_np[y:y+strip_height, :, :] = 0
    return Image.fromarray(img_np)

def add_checkered_strip(img, grid_size=20):
    """Add a black checkerboard pattern to the image."""
    img_np = np.array(img).copy()
    H, W, C = img_np.shape
    x_indices = np.arange(W) // grid_size
    y_indices = np.arange(H) // grid_size
    mask = (y_indices[:, None] + x_indices[None, :]) % 2 == 0
    img_np[mask] = 0
    return Image.fromarray(img_np)


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

    # 5. Gaussian Blur (noise)
    augmented.append(add_blur(img, radius=random.uniform(1.0, 2.5)))

    # 6. Vertical Strip (occlusion)
    augmented.append(add_vertical_strip(img, strip_width_ratio=random.uniform(0.1, 0.2)))

    # 7. Horizontal Strip (occlusion)
    augmented.append(add_horizontal_strip(img, strip_height_ratio=random.uniform(0.1, 0.2)))

    # 8. Checkered Strip (structured noise/occlusion)
    augmented.append(add_checkered_strip(img, grid_size=random.randint(15, 30)))

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