import os
import sys
import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

RANDOM_STATE = 42

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_split_csv, get_split_dataframes, load_dataset, load_dataset_augmented, set_seed


SPLIT_CSV = "data/splits/split.csv"
MODEL_SAVE_PATH = "models/best_logistic_regression.pkl"
EXPERIMENT_NAME = "Plastic_Bottle_Classification"
IMAGE_SIZE = (128, 128)


def train_lr():
    """Train Logistic Regression with augmented data + StandardScaler + GridSearchCV + MLflow."""

    # --- Load data ---
    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("  LOGISTIC REGRESSION - Training")
    print("=" * 60)

    df = read_split_csv(SPLIT_CSV)
    train_df, val_df, test_df = get_split_dataframes(df)

    print(f"\nLoading training data WITH augmentation ({len(train_df)} original images)...")
    X_train, y_train = load_dataset_augmented(train_df, image_size=IMAGE_SIZE)

    print(f"Loading validation data WITHOUT augmentation ({len(val_df)} images)...")
    X_val, y_val = load_dataset(val_df, image_size=IMAGE_SIZE)

    # --- MLflow setup ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LogisticRegression"):
        # --- Pipeline: StandardScaler + LogisticRegression ---
        # StandardScaler normalizes features (mean=0, std=1) for better convergence
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])

        # GridSearchCV searches over classifier params (prefix with 'clf__')
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__solver': ['lbfgs', 'liblinear'],
        }

        print("\nRunning GridSearchCV (with StandardScaler)...")
        grid_search = GridSearchCV(
            pipe,
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

        # --- Evaluate on validation set ---
        y_val_pred = best_model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"\n--- Validation Results ---")
        print(f"F1 Score (weighted): {val_f1:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred, target_names=["others", "plastic_bottle"]))

        # --- Log to MLflow ---
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("augmentation", "flip, rotation, brightness, contrast")
        mlflow.log_param("standardization", "StandardScaler")
        mlflow.log_param("train_samples_augmented", len(X_train))
        for key, value in best_params.items():
            mlflow.log_param(key.replace('clf__', ''), value)
        mlflow.log_param("image_size", IMAGE_SIZE)

        mlflow.log_metric("val_f1_weighted", val_f1)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # --- Save best model (Pipeline includes scaler!) ---
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_SAVE_PATH)
        mlflow.log_artifact(MODEL_SAVE_PATH)
        print(f"\nBest model saved to: {MODEL_SAVE_PATH}")
        print("  (Pipeline includes StandardScaler - no separate scaler needed)")

    return best_model, val_f1


if __name__ == "__main__":
    train_lr()
