"""
Downloads the labeled dataset from Roboflow in YOLOv8 format.

Requires the following in .env (see .env.example):
  ROBOFLOW_API_KEY
  ROBOFLOW_WORKSPACE
  ROBOFLOW_PROJECT
  ROBOFLOW_VERSION

Usage:
    python download_dataset.py
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def main():
    missing = [name for name, val in [
        ("ROBOFLOW_API_KEY", API_KEY),
        ("ROBOFLOW_WORKSPACE", WORKSPACE),
        ("ROBOFLOW_PROJECT", PROJECT),
    ] if not val]
    if missing:
        raise SystemExit(
            f"Missing required .env values: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in first."
        )

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    print(f"Downloading {WORKSPACE}/{PROJECT} v{VERSION} in YOLOv8 format ...")
    dataset = version.download("yolov8", location=DATASET_DIR)

    print(f"Done. Dataset saved to: {dataset.location}")
    print(f"data.yaml should now be at: {os.path.join(dataset.location, 'data.yaml')}")


if __name__ == "__main__":
    main()
