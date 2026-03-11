"""
Master script: Train all 3 models and evaluate them.
Usage: python src/train_all.py
"""
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.models.train_lr import train_lr
from src.models.train_svm import train_svm_model
from src.models.train_cnn import train_cnn
from src.evaluate import evaluate_all


def main():
    print("╔" + "═" * 58 + "╗")
    print("║   PLASTIC BOTTLE CLASSIFICATION - Training Pipeline      ║")
    print("╚" + "═" * 58 + "╝")

    # Check prerequisites
    split_csv = "data/splits/split.csv"
    if not os.path.exists(split_csv):
        print(f"\n❌ Error: '{split_csv}' not found!")
        print("   Please run 'python src/split_data.py' first.")
        sys.exit(1)

    print("\n" + "━" * 60)
    print("  STEP 1/4: Training Logistic Regression")
    print("━" * 60)
    lr_model, lr_f1 = train_lr()

    print("\n" + "━" * 60)
    print("  STEP 2/4: Training SVM")
    print("━" * 60)
    svm_model, svm_f1 = train_svm_model()

    print("\n" + "━" * 60)
    print("  STEP 3/4: Training CNN")
    print("━" * 60)
    cnn_model, cnn_f1 = train_cnn()

    print("\n" + "━" * 60)
    print("  STEP 4/4: Final Evaluation on Test Set")
    print("━" * 60)
    results = evaluate_all()

    print("\n╔" + "═" * 58 + "╗")
    print("║   Pipeline completed! Check MLflow UI:                   ║")
    print("║   Run: mlflow ui                                         ║")
    print("║   Open: http://localhost:5000                             ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
