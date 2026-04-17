from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(r"D:\AUPP\Junior\Spring semester\Machine Learning\finalproject")


def main():
    data_yaml = PROJECT_ROOT / "data" / "dataset" / "dataset.yaml"

    print(f"[INFO] Using dataset config: {data_yaml}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=8,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name="traffic_sign_detector",
        exist_ok=True
    )
    #model.save("my_train_model.h5")
    model.saveweight("file_name")


if __name__ == "__main__":
    main()

    