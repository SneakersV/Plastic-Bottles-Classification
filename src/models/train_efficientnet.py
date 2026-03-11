# EfficientNet-B0 (Transfer Learning) - Plastic Bottle Classification
import os
import sys
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score, classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_split_csv, get_split_dataframes, set_seed, plot_training_history
from src.models.train_cnn import BottleDataset, get_train_transforms, get_val_transforms

RANDOM_STATE = 42

SPLIT_CSV = "data/splits/split.csv"
MODEL_SAVE_PATH = "models/best_efficientnet.pth"
EXPERIMENT_NAME = "Plastic_Bottle_Classification"
IMAGE_SIZE = (128, 128)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4   # Lower LR for fine-tuning pretrained model
NUM_EPOCHS = 15
NUM_CLASSES = 2


# ======================== Model ========================

def build_efficientnet(num_classes=2, freeze_backbone=True):
    """
    Build EfficientNet-B0 with pretrained ImageNet weights.
    Replaces the final classifier for binary classification.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze all layers except the classifier head.
                         This is standard transfer learning (faster, less overfitting).
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze backbone layers (optional, recommended for small datasets)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    # Original: Linear(1280, 1000) -> New: Linear(1280, num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


# ======================== Training ========================

def train_efficientnet():
    """Train EfficientNet-B0 with transfer learning + data augmentation + MLflow."""

    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("  EfficientNet-B0 - Training (Transfer Learning)")
    print("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    df = read_split_csv(SPLIT_CSV)
    train_df, val_df, _ = get_split_dataframes(df)

    # --- Datasets with transforms (reuse CNN transforms: augmentation + ImageNet norm) ---
    train_transform = get_train_transforms(IMAGE_SIZE)
    val_transform = get_val_transforms(IMAGE_SIZE)

    print(f"\nTraining set: {len(train_df)} images (with on-the-fly augmentation)")
    print(f"Validation set: {len(val_df)} images (normalized only)")
    print(f"Pretrained: EfficientNet-B0 (ImageNet weights)")
    print(f"Strategy: Freeze backbone, fine-tune classifier head")

    train_dataset = BottleDataset(train_df, transform=train_transform, image_size=IMAGE_SIZE)
    val_dataset = BottleDataset(val_df, transform=val_transform, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = build_efficientnet(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require grad (classifier head)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler (reduce LR when val_f1 plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # --- MLflow setup ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="EfficientNet-B0"):
        mlflow.log_param("model_type", "EfficientNet-B0 (Transfer Learning)")
        mlflow.log_param("pretrained", "ImageNet")
        mlflow.log_param("freeze_backbone", True)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("device", str(device))
        mlflow.log_param("augmentation", "RandomResizedCrop, HFlip, Rotation, ColorJitter")
        mlflow.log_param("scheduler", "ReduceLROnPlateau (factor=0.5, patience=3)")

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

            # Step scheduler based on val F1
            scheduler.step(val_f1)

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

            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_epoch_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val Acc: {val_accuracy:.4f} | "
                  f"LR: {current_lr:.1e}", end="")

            # ---- Save best model ----
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  ★ Best!", end="")

            print()

        # ---- Plot training history ----
        plot_path = plot_training_history(history, "EfficientNet-B0")
        mlflow.log_artifact(plot_path)

        # ---- Final report ----
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
    train_efficientnet()
