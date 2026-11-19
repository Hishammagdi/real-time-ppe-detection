"""
Evaluates a trained YOLOv8 checkpoint on the validation/test split
and prints mAP, precision, and recall.

Usage:
    python evaluate.py --weights runs/train/custom_detector/weights/best.pt
"""

import argparse
import os
from ultralytics import YOLO

DEFAULT_DATA_YAML = os.path.join(os.path.dirname(__file__), "dataset", "data.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(data=args.data, split=args.split)

    print("\n=== Evaluation Results ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
