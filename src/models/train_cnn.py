# CNN PyTorch - Plastic Bottle Classification (with Data Augmentation)
import os
import sys
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_split_csv, get_split_dataframes, set_seed, plot_training_history

RANDOM_STATE = 42

SPLIT_CSV = "data/splits/split.csv"
MODEL_SAVE_PATH = "models/best_cnn.pth"
EXPERIMENT_NAME = "Plastic_Bottle_Classification"
IMAGE_SIZE = (128, 128)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_CLASSES = 2

# ImageNet normalization stats (standard practice)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ===================== Transforms =====================

def get_train_transforms(image_size=IMAGE_SIZE):
    """
    Training transforms with data augmentation:
    - RandomResizedCrop: random crop & resize (simulates scale variation)
    - RandomHorizontalFlip: 50% chance to flip horizontally
    - RandomRotation: rotate ±15 degrees
    - ColorJitter: random brightness, contrast, saturation, hue changes
    - Normalize: ImageNet mean/std standardization
    """
    return transforms.Compose([
        transforms.Resize((image_size[0] + 12, image_size[1] + 12)),
        transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size=IMAGE_SIZE):
    """
    Validation/Test transforms (no augmentation, just resize + normalize).
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ======================== Dataset ========================

class BottleDataset(Dataset):
    """
    PyTorch Dataset that loads images from file paths on-the-fly
    and applies transforms (augmentation for train, normalize for val/test).
    """

    def __init__(self, dataframe, transform=None, image_size=(128, 128)):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        label = int(row['label'])

        if self.transform:
            img = self.transform(img)
        else:
            # Fallback: simple resize + to tensor
            img = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])(img)

        return img, label


# ======================== CNN Model ========================

class CNN(nn.Module):
    """
    Simple CNN for binary image classification.
    Input: (N, 3, 128, 128) -> Output: (N, 2)
    """

    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 16 channels, 128x128 -> 64x64
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16 -> 32 channels, 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 32 -> 64 channels, 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After 3 pooling layers: 128 / 2^3 = 16 -> Feature map: 64 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# ======================== Training ========================

def train_cnn():
    """Train CNN model with data augmentation + MLflow tracking. Saves best model by val F1."""

    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("  CNN - Training (with Data Augmentation)")
    print("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    df = read_split_csv(SPLIT_CSV)
    train_df, val_df, _ = get_split_dataframes(df)

    # --- Create Datasets with transforms ---
    train_transform = get_train_transforms(IMAGE_SIZE)
    val_transform = get_val_transforms(IMAGE_SIZE)

    print(f"\nTraining set: {len(train_df)} images (with on-the-fly augmentation)")
    print(f"  Augmentations: RandomResizedCrop, HorizontalFlip, Rotation ±15°, ColorJitter")
    print(f"  Normalization: ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD}")
    print(f"Validation set: {len(val_df)} images (no augmentation, normalized only)")

    train_dataset = BottleDataset(train_df, transform=train_transform, image_size=IMAGE_SIZE)
    val_dataset = BottleDataset(val_df, transform=val_transform, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = CNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- MLflow setup ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="CNN"):
        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("device", str(device))
        mlflow.log_param("augmentation", "RandomResizedCrop, HFlip, Rotation, ColorJitter")
        mlflow.log_param("normalization", f"ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD}")

        best_val_f1 = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

        for epoch in range(NUM_EPOCHS):
            # ---- Train phase ----
            model.train()
            running_loss = 0.0
            train_preds_all = []
            train_labels_all = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_preds_all.extend(preds.cpu().numpy())
                train_labels_all.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(train_dataset)
            train_f1 = f1_score(train_labels_all, train_preds_all, average='weighted')

            # ---- Validation phase ----
            model.eval()
            val_preds_all = []
            val_labels_all = []
            val_running_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss_val = criterion(outputs, labels)
                    val_running_loss += loss_val.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_preds_all.extend(preds.cpu().numpy())
                    val_labels_all.extend(labels.cpu().numpy())

            val_epoch_loss = val_running_loss / len(val_dataset)
            val_f1 = f1_score(val_labels_all, val_preds_all, average='weighted')
            val_accuracy = accuracy_score(val_labels_all, val_preds_all)

            # ---- Record history ----
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_epoch_loss)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)

            # ---- Log metrics per epoch ----
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_f1_weighted", val_f1, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_epoch_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}", end="")

            # ---- Save best model ----
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  ★ Best model saved!", end="")

            print()

        # ---- Plot training history ----
        plot_path = plot_training_history(history, "CNN")
        mlflow.log_artifact(plot_path)

        # ---- Final validation report ----
        print(f"\n--- Best Validation F1: {best_val_f1:.4f} ---")
        print("\nFinal Epoch Classification Report (Validation):")
        print(classification_report(
            val_labels_all, val_preds_all,
            target_names=["others", "plastic_bottle"]
        ))

        mlflow.log_metric("best_val_f1_weighted", best_val_f1)
        mlflow.log_artifact(MODEL_SAVE_PATH)

    return model, best_val_f1


if __name__ == "__main__":
    train_cnn()