from typing import Dict, List, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DensityClassifier:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.thresholds = self.config["density_thresholds"]

    def classify(self, vehicle_count: int) -> Dict:
        for level, thresh in self.thresholds.items():
            if thresh["min"] <= vehicle_count <= thresh["max"]:
                return {
                    "vehicle_count": vehicle_count,
                    "density": thresh["label"],
                    "density_level": level,
                    "color": thresh["color"],
                    "threshold_range": f"{thresh['min']}-{thresh['max']}",
                }
        return {
            "vehicle_count": vehicle_count,
            "density": "High",
            "density_level": "high",
            "color": self.thresholds["high"]["color"],
            "threshold_range": f"{self.thresholds['high']['min']}+",
        }

    def classify_from_detections(self, detections: List[Dict]) -> Dict:
        return self.classify(len(detections))

    def classify_with_weights(self, counts: Dict) -> Dict:
        weights = {
            "cars": 1,
            "motorcycles": 0.5,
            "bicycles": 0.3,
            "trucks": 2,
            "buses": 2.5,
        }
        weighted_count = sum(
            counts.get(k, 0) * w for k, w in weights.items()
        )
        return {
            **self.classify(int(weighted_count)),
            "raw_count": counts.get("total_vehicles", 0),
            "weighted_count": round(weighted_count, 2),
        }

    def get_density_trend(self, history: List[Dict]) -> Dict:
        if len(history) < 2:
            return {"trend": "stable", "change": 0}
        recent = history[-1]["vehicle_count"]
        previous = history[-2]["vehicle_count"]
        diff = recent - previous
        trend = "increasing" if diff > 2 else "decreasing" if diff < -2 else "stable"
        return {"trend": trend, "change": diff}

    def update_thresholds(
        self,
        low_max: int = 10,
        medium_max: int = 25,
    ) -> None:
        self.thresholds["low"]["max"] = low_max
        self.thresholds["medium"]["min"] = low_max + 1
        self.thresholds["medium"]["max"] = medium_max
        self.thresholds["high"]["min"] = medium_max + 1
        logger.info(f"Thresholds updated: Low=0-{low_max}, Medium={low_max+1}-{medium_max}, High={medium_max+1}+")
