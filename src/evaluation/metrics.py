import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

matplotlib.use("Agg")

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsVisualizer:
    def __init__(self, plots_dir: str = "reports/plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_training_curves(self, results_csv: str) -> None:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        metrics = {
            "Loss": ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                     "val/box_loss", "val/cls_loss", "val/dfl_loss"],
            "mAP": ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
            "Precision & Recall": ["metrics/precision(B)", "metrics/recall(B)"],
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (title, cols) in zip(axes, metrics.items()):
            for col in cols:
                if col in df.columns:
                    label = col.split("/")[-1].replace("(B)", "")
                    ax.plot(df["epoch"] if "epoch" in df.columns else range(len(df)),
                            df[col], label=label)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)

        plt.tight_layout()
        out = self.plots_dir / "training_curves.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training curves saved: {out}")

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
            ["d", ".2f"],
        ):
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
            )
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        out = self.plots_dir / "confusion_matrix.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved: {out}")

    def plot_pr_curve(
        self,
        precision: List[float],
        recall: List[float],
        class_name: str = "all",
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, "b-", linewidth=2)
        ax.fill_between(recall, precision, alpha=0.1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve ({class_name})", fontweight="bold")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        out = self.plots_dir / f"pr_curve_{class_name}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"PR curve saved: {out}")

    def compare_models(self, nano_metrics: Dict, small_metrics: Dict) -> None:
        metrics_keys = ["precision", "recall", "f1", "map50", "map50_95"]
        labels = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]

        nano_vals = [nano_metrics.get(k, 0) for k in metrics_keys]
        small_vals = [small_metrics.get(k, 0) for k in metrics_keys]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, nano_vals, width, label="YOLOv8n", color="#2196F3", edgecolor="black")
        ax.bar(x + width / 2, small_vals, width, label="YOLOv8s", color="#FF5722", edgecolor="black")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("YOLOv8n vs YOLOv8s — Performance Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(axis="y", alpha=0.5)

        for bars in [ax.containers[0], ax.containers[1]]:
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

        plt.tight_layout()
        out = self.plots_dir / "model_comparison.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Model comparison plot saved: {out}")
