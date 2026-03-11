"""
Unified evaluation script.
Loads the 3 best saved models and evaluates them on the TEST set.
Prints a comparison table of F1 scores.
"""
import os
import sys
import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_split_csv, get_split_dataframes, load_dataset, load_dataset_for_cnn
from src.models.train_cnn import CNN, IMAGE_SIZE, NUM_CLASSES


SPLIT_CSV = "data/splits/split.csv"
TARGET_NAMES = ["others", "plastic_bottle"]

MODELS = {
    "LogisticRegression": "models/best_logistic_regression.pkl",
    "SVM": "models/best_svm.pkl",
    "CNN": "models/best_cnn.pth",
}


def evaluate_sklearn_model(model_path, X_test, y_test):
    """Evaluate a scikit-learn model (LogReg or SVM)."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    return f1, acc, report


def evaluate_cnn_model(model_path, X_test_tensor, y_test_tensor):
    """Evaluate a PyTorch CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(X_test_tensor), batch_size):
            batch_x = X_test_tensor[i:i+batch_size].to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    y_test_np = y_test_tensor.numpy()
    y_pred_np = np.array(all_preds)

    f1 = f1_score(y_test_np, y_pred_np, average='weighted')
    acc = accuracy_score(y_test_np, y_pred_np)
    report = classification_report(y_test_np, y_pred_np, target_names=TARGET_NAMES)
    return f1, acc, report


def evaluate_all():
    """Evaluate all 3 models and print comparison table."""

    print("=" * 60)
    print("  MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    # --- Load test data ---
    df = read_split_csv(SPLIT_CSV)
    _, _, test_df = get_split_dataframes(df)
    print(f"\nTest set size: {len(test_df)} images\n")

    # Flat data for sklearn models
    X_test_flat, y_test_flat = load_dataset(test_df, image_size=IMAGE_SIZE)

    # Tensor data for CNN
    X_test_tensor, y_test_tensor = load_dataset_for_cnn(test_df, image_size=IMAGE_SIZE)

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
            f1, acc, report = evaluate_cnn_model(path, X_test_tensor, y_test_tensor)
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
