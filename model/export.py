"""
Exports a trained YOLOv8 .pt checkpoint to ONNX (optional, for faster inference
in the FastAPI backend later).

Usage:
    python export.py --weights runs/train/custom_detector/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLOv8 model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine"])
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    exported_path = model.export(format=args.format, imgsz=args.imgsz)
    print(f"Exported model saved to: {exported_path}")


if __name__ == "__main__":
    main()
