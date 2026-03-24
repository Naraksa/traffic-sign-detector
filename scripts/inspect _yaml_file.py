from pathlib import Path
import sys


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode("cp1252", errors="replace").decode("cp1252")
        print(safe_text)


def print_file_content(file_path: Path):
    print(f"\n[INFO] Reading: {file_path}\n")
    if not file_path.exists():
        print("[ERROR] File does not exist.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    safe_print(content)


if __name__ == "__main__":
    project_root = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject")
    data_dir = project_root / "data" / "raw" / "external" / "kaggle" / "data"

    files_to_check = [
        data_dir / "Traffic_Sign.yaml",
        data_dir / "Traffic_Sign_class.yaml",
        data_dir / "Traffic_Sign_name.yaml",
    ]

    print(f"[INFO] Using data directory: {data_dir}")

    for file_path in files_to_check:
        print_file_content(file_path)