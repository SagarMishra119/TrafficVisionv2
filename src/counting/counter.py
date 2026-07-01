from collections import Counter
from typing import Dict, List

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VehicleCounter:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.classes = self.config["classes"]

    def count(self, detections: List[Dict]) -> Dict:
        type_counts = Counter(d["vehicle_type"] for d in detections)
        result = {
            "total_vehicles": len(detections),
            "cars": type_counts.get("car", 0),
            "buses": type_counts.get("bus", 0),
            "trucks": type_counts.get("truck", 0),
            "motorcycles": type_counts.get("motorcycle", 0),
            "bicycles": type_counts.get("bicycle", 0),
        }
        return result

    def count_by_confidence(
        self,
        detections: List[Dict],
        min_confidence: float = 0.5,
    ) -> Dict:
        filtered = [d for d in detections if d["confidence"] >= min_confidence]
        return self.count(filtered)

    def aggregate_video_counts(self, frame_counts: List[Dict]) -> Dict:
        if not frame_counts:
            return self.count([])
        keys = ["total_vehicles", "cars", "buses", "trucks", "motorcycles", "bicycles"]
        avg = {k: round(sum(f.get(k, 0) for f in frame_counts) / len(frame_counts), 2) for k in keys}
        peak = {k: max(f.get(k, 0) for f in frame_counts) for k in keys}
        return {
            "average": avg,
            "peak": peak,
            "total_frames": len(frame_counts),
        }

    def get_class_confidence_stats(self, detections: List[Dict]) -> Dict:
        from collections import defaultdict
        conf_by_class = defaultdict(list)
        for d in detections:
            conf_by_class[d["vehicle_type"]].append(d["confidence"])

        stats = {}
        for cls, confs in conf_by_class.items():
            stats[cls] = {
                "count": len(confs),
                "mean_conf": round(sum(confs) / len(confs), 4),
                "max_conf": round(max(confs), 4),
                "min_conf": round(min(confs), 4),
            }
        return stats
