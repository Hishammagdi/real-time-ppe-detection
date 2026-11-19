"""
Fine-tunes a YOLOv8 model on the custom dataset downloaded via download_dataset.py.

Usage:
    python train.py
    python train.py --data dataset/data.yaml --epochs 50
"""

import os
import argparse
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

DEFAULT_DATA_YAML = os.path.join(os.path.dirname(__file__), "dataset", "data.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a custom dataset")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--model", default=os.getenv("BASE_MODEL", "yolov8n.pt"),
                         help="Base pretrained checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 100)))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("IMG_SIZE", 640)))
    parser.add_argument("--batch", type=int, default=int(os.getenv("BATCH_SIZE", 16)))
    parser.add_argument("--name", default=os.getenv("RUN_NAME", "custom_detector"))
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"data.yaml not found at {args.data}. Run download_dataset.py first, "
            f"or pass --data with the correct path."
        )

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project="runs/train",
    )

    best_weights = os.path.join("runs", "train", args.name, "weights", "best.pt")
    print(f"\nTraining complete. Best weights saved to: {best_weights}")
    print("Copy this path into the backend's MODEL_WEIGHTS_PATH env variable.")


if __name__ == "__main__":
    main()
