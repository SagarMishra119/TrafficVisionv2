import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrafficAnalytics:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._session_log: List[Dict] = []

    def log_frame(
        self,
        frame_id: int,
        detections: List[Dict],
        counts: Dict,
        density: Dict,
        timestamp: Optional[str] = None,
    ) -> None:
        entry = {
            "frame_id": frame_id,
            "timestamp": timestamp or datetime.now().isoformat(),
            "total_vehicles": counts.get("total_vehicles", 0),
            "cars": counts.get("cars", 0),
            "buses": counts.get("buses", 0),
            "trucks": counts.get("trucks", 0),
            "motorcycles": counts.get("motorcycles", 0),
            "bicycles": counts.get("bicycles", 0),
            "density": density.get("density", "Unknown"),
        }
        self._session_log.append(entry)

    def get_session_summary(self) -> Dict:
        if not self._session_log:
            return {}
        df = pd.DataFrame(self._session_log)
        return {
            "total_frames": len(df),
            "avg_vehicles": round(df["total_vehicles"].mean(), 2),
            "max_vehicles": int(df["total_vehicles"].max()),
            "min_vehicles": int(df["total_vehicles"].min()),
            "std_vehicles": round(df["total_vehicles"].std(), 2),
            "density_distribution": df["density"].value_counts().to_dict(),
            "avg_by_type": {
                "cars": round(df["cars"].mean(), 2),
                "buses": round(df["buses"].mean(), 2),
                "trucks": round(df["trucks"].mean(), 2),
                "motorcycles": round(df["motorcycles"].mean(), 2),
                "bicycles": round(df["bicycles"].mean(), 2),
            },
            "peak_frame": int(df.loc[df["total_vehicles"].idxmax(), "frame_id"]),
        }

    def export_to_csv(self, filename: str = "session_analytics.csv") -> str:
        output = self.reports_dir / filename
        df = pd.DataFrame(self._session_log)
        df.to_csv(str(output), index=False)
        logger.info(f"Analytics exported: {output}")
        return str(output)

    def export_to_json(self, filename: str = "session_analytics.json") -> str:
        output = self.reports_dir / filename
        with open(output, "w") as f:
            json.dump(self._session_log, f, indent=2)
        logger.info(f"Analytics exported: {output}")
        return str(output)

    def compute_congestion_index(self, counts: Dict) -> float:
        weights = {"cars": 1.0, "buses": 2.5, "trucks": 2.0, "motorcycles": 0.5, "bicycles": 0.3}
        raw = sum(counts.get(k, 0) * w for k, w in weights.items())
        return round(min(raw / 50.0, 1.0) * 100, 2)

    def get_hourly_breakdown(self) -> Dict:
        if not self._session_log:
            return {}
        df = pd.DataFrame(self._session_log)
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        hourly = df.groupby("hour")["total_vehicles"].agg(["mean", "max"]).reset_index()
        return hourly.to_dict(orient="records")

    def clear_session(self) -> None:
        self._session_log = []
        logger.info("Session analytics cleared")

    def get_realtime_stats(self, window: int = 10) -> Dict:
        if not self._session_log:
            return {}
        recent = self._session_log[-window:]
        df = pd.DataFrame(recent)
        return {
            "window_frames": len(recent),
            "current_vehicles": recent[-1]["total_vehicles"],
            "trend_avg": round(df["total_vehicles"].mean(), 2),
            "current_density": recent[-1]["density"],
        }
