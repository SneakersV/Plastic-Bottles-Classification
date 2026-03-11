# CNN PyTorch - Plastic Bottle Classification
import os
import sys
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_split_csv, get_split_dataframes, load_dataset_for_cnn


SPLIT_CSV = "data/splits/split.csv"
MODEL_SAVE_PATH = "models/best_cnn.pth"
EXPERIMENT_NAME = "Plastic_Bottle_Classification"
IMAGE_SIZE = (128, 128)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_CLASSES = 2


# ======================== Dataset ========================

class BottleDataset(Dataset):
    """Simple dataset wrapping pre-loaded tensors."""

    def __init__(self, images, labels):
        self.images = images   # Tensor (N, 3, H, W)
        self.labels = labels   # Tensor (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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
    """Train CNN model with MLflow tracking. Saves best model by val F1."""

    print("=" * 60)
    print("  CNN - Training")
    print("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    df = read_split_csv(SPLIT_CSV)
    train_df, val_df, _ = get_split_dataframes(df)

    print(f"Loading training data ({len(train_df)} images)...")
    X_train, y_train = load_dataset_for_cnn(train_df, image_size=IMAGE_SIZE)

    print(f"Loading validation data ({len(val_df)} images)...")
    X_val, y_val = load_dataset_for_cnn(val_df, image_size=IMAGE_SIZE)

    # --- DataLoaders ---
    train_dataset = BottleDataset(X_train, y_train)
    val_dataset = BottleDataset(X_val, y_val)

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

        best_val_f1 = 0.0

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

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_preds_all.extend(preds.cpu().numpy())
                    val_labels_all.extend(labels.cpu().numpy())

            val_f1 = f1_score(val_labels_all, val_preds_all, average='weighted')
            val_accuracy = accuracy_score(val_labels_all, val_preds_all)

            # ---- Log metrics per epoch ----
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_f1_weighted", val_f1, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                  f"Loss: {epoch_loss:.4f} | "
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