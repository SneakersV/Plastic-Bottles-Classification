"""
Master script: Train selected models and evaluate them.

Usage:
    python src/train_all.py                  # Train ALL models
    python src/train_all.py cnn efnet        # Train only CNN and EfficientNet
    python src/train_all.py lr svm           # Train only LogReg and SVM
    python src/train_all.py cnn              # Train only CNN

Available model names:
    lr       / logreg       -> Logistic Regression
    svm                     -> Support Vector Machine
    cnn                     -> CNN (Custom architecture)
    efnet    / efficientnet -> EfficientNet-B0 (Transfer Learning)
"""
import os
import sys
import argparse

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.models.train_lr import train_lr
from src.models.train_svm import train_svm_model
from src.models.train_cnn import train_cnn
from src.models.train_efficientnet import train_efficientnet
from src.evaluate import evaluate_all

# Map of CLI shorthand -> (display name, train function)
MODEL_REGISTRY = {
    'lr':            ('Logistic Regression', train_lr),
    'logreg':        ('Logistic Regression', train_lr),
    'svm':           ('SVM', train_svm_model),
    'cnn':           ('CNN', train_cnn),
    'efnet':         ('EfficientNet-B0 (Transfer Learning)', train_efficientnet),
    'efficientnet':  ('EfficientNet-B0 (Transfer Learning)', train_efficientnet),
}

# Default order when training all models
ALL_MODELS = ['lr', 'svm', 'cnn', 'efnet']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train plastic bottle classification models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train_all.py                  # Train ALL 4 models
  python src/train_all.py cnn efnet        # Train CNN and EfficientNet only
  python src/train_all.py lr               # Train Logistic Regression only

Available model names:
  lr / logreg      Logistic Regression
  svm              Support Vector Machine
  cnn              CNN (Custom architecture)
  efnet / efficientnet  EfficientNet-B0 (Transfer Learning)
        """
    )
    parser.add_argument(
        'models',
        nargs='*',
        default=[],
        help='Model(s) to train. If empty, trains all models.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which models to train
    if args.models:
        # Validate user input
        selected = []
        seen_functions = set()
        for name in args.models:
            name_lower = name.lower()
            if name_lower not in MODEL_REGISTRY:
                print(f"❌ Unknown model: '{name}'")
                print(f"   Available: {', '.join(sorted(set(MODEL_REGISTRY.keys())))}")
                sys.exit(1)
            display_name, func = MODEL_REGISTRY[name_lower]
            # Avoid duplicates (e.g., 'lr' and 'logreg' point to same function)
            if func not in seen_functions:
                selected.append((name_lower, display_name, func))
                seen_functions.add(func)
    else:
        selected = [(key, MODEL_REGISTRY[key][0], MODEL_REGISTRY[key][1]) for key in ALL_MODELS]

    total_steps = len(selected) + 1  # +1 for evaluation

    print("╔" + "═" * 58 + "╗")
    print("║   PLASTIC BOTTLE CLASSIFICATION - Training Pipeline      ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Models to train: {', '.join(d for _, d, _ in selected)}")

    # Check prerequisites
    split_csv = "data/splits/split.csv"
    if not os.path.exists(split_csv):
        print(f"\n❌ Error: '{split_csv}' not found!")
        print("   Please run 'python src/split_data.py' first.")
        sys.exit(1)

    # Train each selected model
    results = {}
    for i, (key, display_name, train_func) in enumerate(selected, 1):
        print("\n" + "━" * 60)
        print(f"  STEP {i}/{total_steps}: Training {display_name}")
        print("━" * 60)
        model, f1 = train_func()
        results[display_name] = f1

    # Evaluate all available models (including previously trained ones)
    print("\n" + "━" * 60)
    print(f"  STEP {total_steps}/{total_steps}: Final Evaluation on Test Set")
    print("━" * 60)
    eval_results = evaluate_all()

    print("\n╔" + "═" * 58 + "╗")
    print("║   Pipeline completed! Check MLflow UI:                   ║")
    print("║   Run: mlflow ui                                         ║")
    print("║   Open: http://localhost:5000                             ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
