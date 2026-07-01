from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrafficVisualizer:
    DENSITY_COLORS = {
        "Low": "#00C853",
        "Medium": "#FFD600",
        "High": "#D50000",
    }

    def __init__(self, plots_dir: str = "reports/plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_vehicle_timeline(self, session_log: List[Dict], filename: str = "vehicle_timeline.png") -> str:
        if not session_log:
            return ""
        df = pd.DataFrame(session_log)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        axes[0].plot(df["frame_id"], df["total_vehicles"], "b-", linewidth=1.5, label="Total")
        axes[0].fill_between(df["frame_id"], df["total_vehicles"], alpha=0.15, color="blue")
        axes[0].set_title("Vehicle Count Over Time", fontweight="bold")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        vehicle_types = ["cars", "buses", "trucks", "motorcycles", "bicycles"]
        colors = ["#2196F3", "#FF5722", "#9C27B0", "#00BCD4", "#8BC34A"]
        for vtype, color in zip(vehicle_types, colors):
            if vtype in df.columns:
                axes[1].plot(df["frame_id"], df[vtype], label=vtype.capitalize(), color=color, linewidth=1)
        axes[1].set_title("Vehicle Types Over Time", fontweight="bold")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Count")
        axes[1].legend(ncol=3)

        plt.tight_layout()
        out = self.plots_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Timeline saved: {out}")
        return str(out)

    def plot_density_distribution(self, session_log: List[Dict], filename: str = "density_distribution.png") -> str:
        if not session_log:
            return ""
        df = pd.DataFrame(session_log)
        dist = df["density"].value_counts()
        colors = [self.DENSITY_COLORS.get(k, "#888888") for k in dist.index]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(dist.index, dist.values, color=colors, edgecolor="black")
        axes[0].set_title("Density Level Distribution", fontweight="bold")
        axes[0].set_ylabel("Frame Count")

        axes[1].pie(dist.values, labels=dist.index, colors=colors, autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Density Level %", fontweight="bold")

        plt.tight_layout()
        out = self.plots_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        return str(out)

    def plot_vehicle_composition(self, counts: Dict, filename: str = "vehicle_composition.png") -> str:
        vehicle_types = ["cars", "buses", "trucks", "motorcycles", "bicycles"]
        values = [counts.get(k, 0) for k in vehicle_types]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(vehicle_types, values, color=["#2196F3", "#FF5722", "#9C27B0", "#00BCD4", "#8BC34A"])
        ax.set_title("Vehicle Composition", fontweight="bold")
        ax.set_xlabel("Count")
        ax.bar_label(bars, padding=3)
        plt.tight_layout()
        out = self.plots_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        return str(out)

    def plot_summary_dashboard(self, summary: Dict, filename: str = "dashboard_summary.png") -> str:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Traffic Analysis Dashboard", fontsize=16, fontweight="bold")

        ax1 = fig.add_subplot(2, 2, 1)
        stats_text = (
            f"Total Frames: {summary.get('total_frames', 0)}\n"
            f"Avg Vehicles: {summary.get('avg_vehicles', 0)}\n"
            f"Peak Vehicles: {summary.get('max_vehicles', 0)}\n"
            f"Std Dev: {summary.get('std_vehicles', 0)}"
        )
        ax1.text(0.1, 0.5, stats_text, transform=ax1.transAxes, fontsize=12,
                 verticalalignment="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
        ax1.set_title("Session Statistics", fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 2, 2)
        density_dist = summary.get("density_distribution", {})
        if density_dist:
            colors = [self.DENSITY_COLORS.get(k, "#888888") for k in density_dist.keys()]
            ax2.pie(density_dist.values(), labels=density_dist.keys(), colors=colors, autopct="%1.1f%%")
        ax2.set_title("Density Distribution", fontweight="bold")

        ax3 = fig.add_subplot(2, 2, 3)
        avg_by_type = summary.get("avg_by_type", {})
        if avg_by_type:
            ax3.bar(avg_by_type.keys(), avg_by_type.values(), color="#2196F3", edgecolor="black")
            ax3.set_title("Average Vehicle Types", fontweight="bold")
            ax3.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        out = self.plots_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Dashboard summary saved: {out}")
        return str(out)
