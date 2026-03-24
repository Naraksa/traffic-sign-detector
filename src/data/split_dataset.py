"""
We will only use image-label pair, so we will skip the 9 unlabel images.
=> 2757 image-label pair

train_test_split:
- 70% train - 1929
- 20% val - 551
- 10% test - 277
"""


import random
import shutil
from pathlib import Path


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_paired_files(images_dir: Path, labels_dir: Path):
    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_exts])

    pairs = []
    missing_labels = []

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path)

    return pairs, missing_labels


def split_dataset(pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


def copy_pairs(pairs, image_out_dir: Path, label_out_dir: Path):
    make_dir(image_out_dir)
    make_dir(label_out_dir)

    for img_path, label_path in pairs:
        shutil.copy2(img_path, image_out_dir / img_path.name)
        shutil.copy2(label_path, label_out_dir / label_path.name)


def write_dataset_yaml(dataset_root: Path, class_names=None):
    yaml_path = dataset_root / "dataset.yaml"

    if class_names is None:
        class_names = ["Traffic Sign"]

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"[INFO] dataset.yaml written to: {yaml_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    raw_images_dir = project_root / "data" / "raw" / "external" / "kaggle" / "images"
    raw_labels_dir = project_root / "data" / "raw" / "external" / "kaggle" / "labels"

    dataset_root = project_root / "data" / "dataset"

    image_train_dir = dataset_root / "images" / "train"
    image_val_dir = dataset_root / "images" / "val"
    image_test_dir = dataset_root / "images" / "test"

    label_train_dir = dataset_root / "labels" / "train"
    label_val_dir = dataset_root / "labels" / "val"
    label_test_dir = dataset_root / "labels" / "test"

    # Clear existing dataset output
    clear_directory(dataset_root / "images")
    clear_directory(dataset_root / "labels")

    # Get valid image-label pairs
    pairs, missing_labels = get_paired_files(raw_images_dir, raw_labels_dir)

    print(f"[INFO] Valid pairs found: {len(pairs)}")
    print(f"[INFO] Images skipped (missing labels): {len(missing_labels)}")

    if missing_labels:
        print("[SAMPLE] Skipped images:")
        for img in missing_labels[:10]:
            print(f" - {img.name}")

    # Split
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    print(f"[INFO] Train pairs: {len(train_pairs)}")
    print(f"[INFO] Val pairs: {len(val_pairs)}")
    print(f"[INFO] Test pairs: {len(test_pairs)}")

    # Copy files
    copy_pairs(train_pairs, image_train_dir, label_train_dir)
    copy_pairs(val_pairs, image_val_dir, label_val_dir)
    copy_pairs(test_pairs, image_test_dir, label_test_dir)

    print("[INFO] Dataset split and copied successfully.")

    # TODO: replace with real class names after inspecting YAML/class file
    class_names = []

    write_dataset_yaml(dataset_root, class_names=class_names)