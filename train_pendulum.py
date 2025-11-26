from ultralytics import YOLO

def define_yolo_model():
    # 1. Load a small pre-trained YOLO model (good for edge devices)
    model = YOLO("yolo11n.pt")  # or "yolov8n.pt" depending on version

    # 2. Train on your dataset
    results = model.train(
        data="pendulum_dataset\pendulum.yaml",  # path to dataset YAML
        epochs=100,           # try 30-100
        imgsz=640,           # image size
        batch=16,            # reduce if running out of memory
        project="runs_pendulum",
        name="pendulum_yolo",
    )

    # run training
    return results

def main():
    define_yolo_model()

if __name__ == "__main__":
    main()