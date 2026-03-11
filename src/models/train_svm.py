import os
import sys
import joblib
import mlflow
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_split_csv, get_split_dataframes, load_dataset


SPLIT_CSV = "data/splits/split.csv"
MODEL_SAVE_PATH = "models/best_svm.pkl"
EXPERIMENT_NAME = "Plastic_Bottle_Classification"
IMAGE_SIZE = (128, 128)


def train_svm_model():
    """Train SVM with GridSearchCV + MLflow tracking."""

    # --- Load data ---
    print("=" * 60)
    print("  SVM - Training")
    print("=" * 60)

    df = read_split_csv(SPLIT_CSV)
    train_df, val_df, test_df = get_split_dataframes(df)

    print(f"Loading training data ({len(train_df)} images)...")
    X_train, y_train = load_dataset(train_df, image_size=IMAGE_SIZE)

    print(f"Loading validation data ({len(val_df)} images)...")
    X_val, y_val = load_dataset(val_df, image_size=IMAGE_SIZE)

    # --- MLflow setup ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="SVM"):
        # --- GridSearchCV for best hyperparameters ---
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
        }

        print("\nRunning GridSearchCV...")
        grid_search = GridSearchCV(
            svm.SVC(),
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"\nBest params: {best_params}")

        # --- Evaluate on validation set (NO re-fitting!) ---
        y_val_pred = best_model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"\n--- Validation Results ---")
        print(f"F1 Score (weighted): {val_f1:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred, target_names=["others", "plastic_bottle"]))

        # --- Log to MLflow ---
        mlflow.log_param("model_type", "SVM")
        for key, value in best_params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("image_size", IMAGE_SIZE)

        mlflow.log_metric("val_f1_weighted", val_f1)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # --- Save best model ---
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_SAVE_PATH)
        mlflow.log_artifact(MODEL_SAVE_PATH)
        print(f"\nBest model saved to: {MODEL_SAVE_PATH}")

    return best_model, val_f1


if __name__ == "__main__":
    train_svm_model()