import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.utils.config import load_config
from src.utils.file_utils import get_image_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetStats:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.dataset_path = Path(self.config["paths"]["dataset"])
        self.classes = self.config["classes"]
        self.plots_dir = Path(self.config["paths"]["plots"])
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def count_annotations(self, split: str) -> Dict[str, int]:
        lbl_dir = self.dataset_path / split / "labels"
        class_counts = Counter()
        if not lbl_dir.exists():
            return {}
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_counts[int(parts[0])] += 1
                        except (ValueError, IndexError):
                            pass
        return {self.classes[k]: v for k, v in class_counts.items() if k < len(self.classes)}

    def generate_full_report(self) -> Dict:
        report = {"splits": {}, "total_images": 0, "class_distribution": Counter()}
        for split in ["train", "valid", "test"]:
            img_dir = self.dataset_path / split / "images"
            imgs = get_image_paths(str(img_dir)) if img_dir.exists() else []
            annotations = self.count_annotations(split)
            report["splits"][split] = {
                "images": len(imgs),
                "annotations": annotations,
            }
            report["total_images"] += len(imgs)
            for cls, cnt in annotations.items():
                report["class_distribution"][cls] += cnt

        report["class_distribution"] = dict(report["class_distribution"])
        logger.info(f"Total images: {report['total_images']}")
        logger.info(f"Class distribution: {report['class_distribution']}")
        return report

    def save_report(self, report: Dict, output_path: str = "reports/dataset_stats.json") -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Dataset stats saved: {output_path}")

    def plot_class_distribution(self, report: Dict) -> None:
        dist = report.get("class_distribution", {})
        if not dist:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        classes = list(dist.keys())
        counts = list(dist.values())

        axes[0].bar(classes, counts, color="#2196F3", edgecolor="black")
        axes[0].set_title("Class Distribution", fontweight="bold")
        axes[0].set_xlabel("Vehicle Class")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=15)

        axes[1].pie(counts, labels=classes, autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Class Distribution %", fontweight="bold")

        plt.tight_layout()
        out = self.plots_dir / "class_distribution.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Class distribution plot saved: {out}")

    def plot_split_distribution(self, report: Dict) -> None:
        splits = list(report["splits"].keys())
        counts = [report["splits"][s]["images"] for s in splits]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(splits, counts, color=["#4CAF50", "#2196F3", "#FF5722"], edgecolor="black")
        ax.set_title("Train / Valid / Test Split", fontweight="bold")
        ax.set_xlabel("Split")
        ax.set_ylabel("Number of Images")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center")
        plt.tight_layout()
        out = self.plots_dir / "split_distribution.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Split distribution plot saved: {out}")

    def run(self) -> Dict:
        report = self.generate_full_report()
        self.save_report(report)
        self.plot_class_distribution(report)
        self.plot_split_distribution(report)
        return report
