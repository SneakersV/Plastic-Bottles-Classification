"""
Unified evaluation script.
Loads all best saved models and evaluates them on the TEST set.
Prints a comparison table of F1 scores.
"""
import os
import sys
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_split_csv, get_split_dataframes, load_dataset, config
from src.models.train_cnn import CNN, BottleDataset, get_val_transforms, IMAGE_SIZE, NUM_CLASSES
from src.models.train_efficientnet import build_efficientnet


SPLIT_CSV = config["paths"]["split_csv"]
TARGET_NAMES = config["classes"]["names"]

MODELS = {
    "LogisticRegression": os.path.join(config["paths"]["model_save_dir"], "best_logistic_regression.pkl"),
    "SVM": os.path.join(config["paths"]["model_save_dir"], "best_svm.pkl"),
    "CNN": os.path.join(config["paths"]["model_save_dir"], "best_cnn.pth"),
    "EfficientNet-B0": os.path.join(config["paths"]["model_save_dir"], "best_efficientnet.pth"),
}


def evaluate_sklearn_model(model_path, X_test, y_test):
    """Evaluate a scikit-learn Pipeline model (scaler + classifier)."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    return f1, acc, report


def evaluate_cnn_model(model_path, test_df):
    """Evaluate a PyTorch CNN model using val transforms (with ImageNet normalization)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same val transforms (resize + normalize, NO augmentation)
    val_transform = get_val_transforms(IMAGE_SIZE)
    test_dataset = BottleDataset(test_df, transform=val_transform, image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_test_np = np.array(all_labels)
    y_pred_np = np.array(all_preds)

    f1 = f1_score(y_test_np, y_pred_np, average='weighted')
    acc = accuracy_score(y_test_np, y_pred_np)
    report = classification_report(y_test_np, y_pred_np, target_names=TARGET_NAMES)
    return f1, acc, report


def evaluate_efficientnet_model(model_path, test_df):
    """Evaluate EfficientNet-B0 model using val transforms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = get_val_transforms(IMAGE_SIZE)
    test_dataset = BottleDataset(test_df, transform=val_transform, image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = build_efficientnet(num_classes=NUM_CLASSES, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_test_np = np.array(all_labels)
    y_pred_np = np.array(all_preds)

    f1 = f1_score(y_test_np, y_pred_np, average='weighted')
    acc = accuracy_score(y_test_np, y_pred_np)
    report = classification_report(y_test_np, y_pred_np, target_names=TARGET_NAMES)
    return f1, acc, report


def evaluate_all():
    """Evaluate all models and print comparison table."""

    print("=" * 60)
    print("  MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    # --- Load test data ---
    df = read_split_csv(SPLIT_CSV)
    _, _, test_df = get_split_dataframes(df)
    print(f"\nTest set size: {len(test_df)} images\n")

    # Flat data for sklearn models (no augmentation at test time)
    # Note: saved sklearn models include StandardScaler in the Pipeline
    X_test_flat, y_test_flat = load_dataset(test_df, image_size=IMAGE_SIZE)

    results = {}

    # --- Evaluate each model ---
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"⚠  {name}: Model file not found at '{path}'. Skipping.")
            continue

        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")

        if name == "CNN":
            f1, acc, report = evaluate_cnn_model(path, test_df)
        elif name == "EfficientNet-B0":
            f1, acc, report = evaluate_efficientnet_model(path, test_df)
        else:
            f1, acc, report = evaluate_sklearn_model(path, X_test_flat, y_test_flat)

        results[name] = {"f1": f1, "accuracy": acc}
        print(report)

    # --- Summary comparison table ---
    if results:
        print("\n" + "=" * 60)
        print("  COMPARISON TABLE (Test Set)")
        print("=" * 60)
        print(f"{'Model':<25} {'F1 (weighted)':<18} {'Accuracy':<12}")
        print("─" * 55)

        best_name = max(results, key=lambda k: results[k]["f1"])

        for name, metrics in results.items():
            marker = " ★ BEST" if name == best_name else ""
            print(f"{name:<25} {metrics['f1']:<18.4f} {metrics['accuracy']:<12.4f}{marker}")

        print("─" * 55)
        print(f"\n🏆 Best model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

    return results


if __name__ == "__main__":
    evaluate_all()
