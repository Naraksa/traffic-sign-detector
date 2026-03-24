#check if:
# image has a matching .txt label
# labels are in YOLO format
# class IDs are valid
# no empty/corrupt files break traning

#so we will check:
#total number of images
#total number of labels
#images that do not have labels
#labels that do not have images 
#label files follow YOLO format


# from pathlib import Path

# def inspect_data(images_dir: Path, labels_dir: Path, sample_size: int = 20):
#     image_exts = {".jpg", ".jpeg", ".png"}
#     image_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts])
#     label_files = sorted(labels_dir.rglob("*.txt"))

#     print(f"[INFO] Images found: {len(image_files)}")
#     print(f"[INFO] labels found: {len(label_files)}")
    
#     image_stems = {p.stem for p in image_files}
#     label_stems = {p.stem for p in label_files}

#     missing_labels = sorted(image_stems - label_stems)
#     missing_images = sorted(label_stems - image_stems)

#     print(f"[INFO] Images without labels: {len(missing_labels)}")
#     print(f"[INFO] Labels without images: {len(missing_images)}")

#     if missing_labels[:sample_size]:
#         print("\n[SAMPLE] Images without labels:")
#         for name in missing_labels[:sample_size]:
#             print(" -", name)

#     if missing_images[:sample_size]:
#         print("\n[SAMPLE] Labels without images:")
#         for name in missing_images[:sample_size]:
#             print(" -", name)

# def validate_yolo_labels(labels_dir: Path, max_preview: int = 20):
#     label_files = sorted(labels_dir.rglob("*.txt"))
#     invalid_files = []

#     for label_file in label_files:
#         try:
#             with open(label_file, "r", encoding="utf-8") as f:
#                 lines = [line.strip() for line in f if line.strip()]

#             for line in lines:
#                 parts = line.split()
#                 if len(parts) != 5:
#                     invalid_files.append((label_file, f"Wrong column count: {line}"))
#                     break

#                 class_id, x_center, y_center, width, height = parts

#                 int(class_id)
#                 x_center = float(x_center)
#                 y_center = float(y_center)
#                 width = float(width)
#                 height = float(height)

#                 for val in [x_center, y_center, width, height]:
#                     if not (0 <= val <= 1):
#                         invalid_files.append((label_file, f"Out of range: {line}"))
#                         break

#         except Exception as e:
#             invalid_files.append((label_file, str(e)))

#     print(f"\n[INFO] Invalid label files: {len(invalid_files)}")
#     for file_path, reason in invalid_files[:max_preview]:
#         print(f" - {file_path.name}: {reason}")


# if __name__ == "__main__":
#     project_root = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject")

#     images_dir = project_root / "data" / "raw" / "external" / "kaggle" / "images"
#     labels_dir = project_root / "data" / "raw" / "external" / "kaggle" / "labels"
#     inspect_data(images_dir, labels_dir)
#     validate_yolo_labels(labels_dir)

#after inspection, there exist:
"""
- total number of images: 2766
- total number of label images: 2757
- images without labels: 9
- labels without images: 0
- invalid label files: 0

Since there are only 9 images without data annotatoin, we can just drop it.
------ Great dataset ------

"""

import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject")


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


def write_dataset_yaml(dataset_root: Path, class_names):
    yaml_path = dataset_root / "dataset.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

    print(f"[INFO] dataset.yaml written to: {yaml_path}")


if __name__ == "__main__":
    raw_images_dir = PROJECT_ROOT / "data" / "raw" / "external" / "kaggle" / "images"
    raw_labels_dir = PROJECT_ROOT / "data" / "raw" / "external" / "kaggle" / "labels"

    dataset_root = PROJECT_ROOT / "data" / "dataset"

    image_train_dir = dataset_root / "images" / "train"
    image_val_dir = dataset_root / "images" / "val"
    image_test_dir = dataset_root / "images" / "test"

    label_train_dir = dataset_root / "labels" / "train"
    label_val_dir = dataset_root / "labels" / "val"
    label_test_dir = dataset_root / "labels" / "test"

    clear_directory(dataset_root / "images")
    clear_directory(dataset_root / "labels")

    pairs, missing_labels = get_paired_files(raw_images_dir, raw_labels_dir)

    print(f"[INFO] Valid pairs found: {len(pairs)}")
    print(f"[INFO] Images skipped (missing labels): {len(missing_labels)}")

    if missing_labels:
        print("[SAMPLE] Skipped images:")
        for img in missing_labels[:10]:
            print(f" - {img.name}")

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

    copy_pairs(train_pairs, image_train_dir, label_train_dir)
    copy_pairs(val_pairs, image_val_dir, label_val_dir)
    copy_pairs(test_pairs, image_test_dir, label_test_dir)

    print("[INFO] Dataset split and copied successfully.")

    class_names = ["Traffic Sign"]
    write_dataset_yaml(dataset_root, class_names)

    print("[INFO] YOLO dataset preparation completed successfully.")
