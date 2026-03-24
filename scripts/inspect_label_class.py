from pathlib import Path
from collections import Counter


def inspect_class_ids(labels_dir: Path):
    class_counter = Counter()
    invalid_lines = []
    total_boxes = 0
    total_files = 0

    for label_file in sorted(labels_dir.rglob("*.txt")):
        total_files += 1
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                invalid_lines.append((label_file.name, line))
                continue

            try:
                class_id = int(parts[0])
                class_counter[class_id] += 1
                total_boxes += 1
            except ValueError:
                invalid_lines.append((label_file.name, line))

    print(f"[INFO] Label files scanned: {total_files}")
    print(f"[INFO] Total bounding boxes: {total_boxes}")
    print(f"[INFO] Unique class IDs found: {sorted(class_counter.keys())}")

    print("\n[INFO] Class distribution:")
    for class_id in sorted(class_counter.keys()):
        print(f" - Class {class_id}: {class_counter[class_id]} boxes")

    print(f"\n[INFO] Invalid lines: {len(invalid_lines)}")
    if invalid_lines[:10]:
        print("[SAMPLE] Invalid lines:")
        for file_name, line in invalid_lines[:10]:
            print(f" - {file_name}: {line}")


if __name__ == "__main__":
    project_root = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject")
    labels_dir = project_root / "data" / "raw" / "external" / "kaggle" / "labels"

    print(f"[INFO] Inspecting labels in: {labels_dir}")
    inspect_class_ids(labels_dir)

#unique class id is found: [0], so the dataset is currently a single-class object detection dataset.