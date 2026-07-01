from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.plots_dir = Path(self.config["paths"]["plots"])
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        weights_path: str,
        dataset_yaml: str = "configs/dataset.yaml",
        split: str = "test",
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict:
        model = YOLO(weights_path)
        logger.info(f"Evaluating: {weights_path} on [{split}] split")

        metrics = model.val(
            data=dataset_yaml,
            split=split,
            conf=conf,
            iou=iou,
            verbose=True,
        )

        results = self._extract_metrics(metrics)
        logger.info(f"mAP@0.5: {results['map50']:.4f} | mAP@0.5:0.95: {results['map50_95']:.4f}")
        return results

    def _extract_metrics(self, metrics) -> Dict:
        try:
            box = metrics.box
            return {
                "precision": float(box.mp),
                "recall": float(box.mr),
                "f1": float(box.f1.mean()) if hasattr(box, "f1") else 0.0,
                "map50": float(box.map50),
                "map50_95": float(box.map),
                "per_class": {
                    name: {
                        "precision": float(p),
                        "recall": float(r),
                        "ap50": float(ap50),
                        "ap": float(ap),
                    }
                    for name, p, r, ap50, ap in zip(
                        metrics.names.values(),
                        box.p,
                        box.r,
                        box.ap50,
                        box.ap,
                    )
                },
            }
        except Exception as e:
            logger.error(f"Metric extraction error: {e}")
            return {}

    def benchmark_speed(self, weights_path: str, image_size: int = 640) -> Dict:
        model = YOLO(weights_path)
        benchmark_results = model.benchmark(imgsz=image_size)
        return benchmark_results if isinstance(benchmark_results, dict) else {}
