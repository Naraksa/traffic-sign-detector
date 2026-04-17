from pathlib import Path

dataset_path = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject\data\raw\external\kaggle")


def inspect_dataset(root: Path) -> None:
    root = root.resolve()
    print(f"[INFO] Inspecting: {root}")

    if not root.exists():
        print("[ERROR] Path does not exist.")
        return

    items = list(root.rglob("*"))

    if not items:
        print("[WARNING] Folder exists but is empty.")
        return

    print(f"[INFO] Found {len(items)} items. Showing first 50:\n")
    for path in items[:50]:
        print(path.relative_to(root))


if __name__ == "__main__":
    inspect_dataset(dataset_path)

    