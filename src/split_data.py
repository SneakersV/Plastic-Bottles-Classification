import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "./data/raw"

CLASS_MAP = {
    "plastic bottle": 1,
    "others": 0
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

rows = []

for class_folder, label in CLASS_MAP.items():
    class_dir = os.path.join(RAW_DIR, class_folder)

    if not os.path.isdir(class_dir):
        print(f"Warning: folder not found -> {class_dir}")
        continue

    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if os.path.isfile(filepath) and ext in VALID_EXTENSIONS:
            rows.append({
                "filepath": filepath,
                "label": label,
                "class_name": "plastic_bottle" if label == 1 else "others"
            })

df = pd.DataFrame(rows)

print("Total images:", len(df))
print(df["class_name"].value_counts())

# Split: train 70%, val 15%, test 15%
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(split_df["split"].value_counts())
print(split_df.groupby(["split", "class_name"]).size())

os.makedirs("data/splits", exist_ok=True)
split_df.to_csv("data/splits/split.csv", index=False)

print("Saved to data/splits/split.csv")