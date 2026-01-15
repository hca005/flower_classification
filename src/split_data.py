import argparse
from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(data_dir: Path):
    """
    Expect structure:
    data_dir/
      class1/*.jpg
      class2/*.jpg
    Return dataframe with columns: path, label
    """
    rows = []
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(img_path.as_posix()), "label": label})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing class subfolders")
    parser.add_argument("--out_dir", type=str, default="splits", help="Output folder for csv splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = list_images(data_dir)
    if df.empty:
        raise ValueError(f"No images found in: {data_dir}. Expected class subfolders with images.")

    # Stratified split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        shuffle=True,
        stratify=df["label"]
    )

    # Split temp into val and test
    val_size = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size),
        random_state=args.seed,
        shuffle=True,
        stratify=temp_df["label"]
    )

    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("✅ Split done:")
    print(f"Train: {len(train_df)} -> {train_csv}")
    print(f"Val:   {len(val_df)} -> {val_csv}")
    print(f"Test:  {len(test_df)} -> {test_csv}")

    # Quick label counts
    print("\nClass distribution (train top 10):")
    print(train_df["label"].value_counts().head(10))

if __name__ == "__main__":
    main()
