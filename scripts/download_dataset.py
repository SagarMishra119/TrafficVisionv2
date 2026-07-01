"""
Dataset: UA-DETRAC (recommended) or COCO traffic subset via Roboflow.
Run this script to download and prepare the dataset.

Why UA-DETRAC?
  - 100 traffic sequences, 140k+ annotated frames
  - Real-world highway and urban scenarios
  - Multi-class: car, bus, truck (extendable to motorcycle, bicycle)
  - YOLO-compatible conversion available
  - Widely used benchmark — strong baselines exist

Alternative: Roboflow "Traffic Detection" dataset (vehicle-detection-at-any-angle)
  - ~3000+ images, pre-labeled in YOLO format
  - All 5 classes: car, bus, truck, motorcycle, bicycle
  - Ready to use in under 5 minutes
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

ROBOFLOW_WORKSPACE = "vehicle-detection"
ROBOFLOW_PROJECT   = "vehicle-detection-at-any-angle"
ROBOFLOW_VERSION   = 1


def download_via_roboflow(api_key: str, output_dir: str = "dataset") -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=output_dir)
    print(f"Dataset downloaded to: {dataset.location}")
    _restructure_roboflow(dataset.location, output_dir)


def _restructure_roboflow(src: str, dst: str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    for split in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            (dst_path / split / subdir).mkdir(parents=True, exist_ok=True)
            src_dir = src_path / split / subdir
            if src_dir.exists():
                for f in src_dir.iterdir():
                    shutil.copy2(f, dst_path / split / subdir / f.name)
    print("Dataset restructured to project layout.")


def download_sample_dataset(output_dir: str = "dataset") -> None:
    """Creates a minimal synthetic dataset for testing without internet access."""
    import numpy as np
    import cv2

    print("Creating synthetic dataset for testing...")
    classes = ["car", "bus", "truck", "motorcycle", "bicycle"]

    for split in ["train", "valid", "test"]:
        img_dir = Path(output_dir) / split / "images"
        lbl_dir = Path(output_dir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n = 50 if split == "train" else 15
        for i in range(n):
            img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)

            with open(lbl_dir / f"img_{i:04d}.txt", "w") as f:
                num_objects = np.random.randint(1, 6)
                for _ in range(num_objects):
                    cls = np.random.randint(0, len(classes))
                    cx = round(np.random.uniform(0.1, 0.9), 4)
                    cy = round(np.random.uniform(0.1, 0.9), 4)
                    w  = round(np.random.uniform(0.05, 0.3), 4)
                    h  = round(np.random.uniform(0.05, 0.3), 4)
                    f.write(f"{cls} {cx} {cy} {w} {h}\n")

    print(f"Synthetic dataset created in '{output_dir}/'")
    print("  train: 50 images | valid: 15 images | test: 15 images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default="", help="Roboflow API key")
    parser.add_argument("--output", type=str, default="dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data (no internet needed)")
    args = parser.parse_args()

    if args.synthetic:
        download_sample_dataset(args.output)
    elif args.api_key:
        download_via_roboflow(args.api_key, args.output)
    else:
        print("Usage:")
        print("  python scripts/download_dataset.py --api-key YOUR_KEY")
        print("  python scripts/download_dataset.py --synthetic")
