import kagglehub
from pathlib import Path
import shutil

dataset = "sophalratitya/cambodia-traffic-signs-dataset"
source = Path(r"C:\Users\ASUS\.cache\kagglehub\datasets\sophalratitya\cambodia-traffic-signs-dataset\versions\1")
target_kaggle_dir = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject\data\raw\external\kaggle")

def download_kaggle_dataset() -> Path:
    target_kaggle_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(
        kagglehub.dataset_download(
            dataset,
            output_dir=str(target_kaggle_dir),
            force_download=False
        )
    )

    print(f"[INFO] Dataset downloaded to: {downloaded_path.resolve()}")
    return downloaded_path

def import_to_project():
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    
    target_kaggle_dir.mkdir(parents=True, exist_ok=True)

    final_target = target_kaggle_dir / source.name

    if final_target.exists():
        print(f"[INFO] Target already exists: {final_target}")
        return

    shutil.copytree(source, final_target)
    print(f"[INFO] Copied dataset to: {final_target.resolve()}")

def main():
    downloaded_path = Path(kagglehub.dataset_download(dataset))
    print("[INFO] KaggleHub cache path:")
    print(downloaded_path.resolve())
    print("[INFO] Exists:", downloaded_path.exists())


if __name__ == "__main__":
    import_to_project()


