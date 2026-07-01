import json
from pathlib import Path
from typing import Dict, Optional

from src.data.dataset_manager import DatasetManager
from src.data.dataset_validator import DatasetValidator
from src.data.dataset_stats import DatasetStats
from src.training.trainer import YOLOTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.config_path = config_path
        setup_logger()

    def step_validate_dataset(self) -> Dict:
        logger.info("=== Step: Validating Dataset ===")
        validator = DatasetValidator(self.config["paths"]["dataset"])
        report = validator.validate_all()
        self._save_json(report, "reports/validation_report.json")
        return report

    def step_generate_stats(self) -> Dict:
        logger.info("=== Step: Generating Dataset Statistics ===")
        stats = DatasetStats(self.config_path)
        return stats.run()

    def step_train_variant(self, variant: str, dataset_yaml: str = "configs/dataset.yaml") -> str:
        labels = {
            "v8n": "YOLOv8n (Baseline)", "v8s": "YOLOv8s (Improved)",
            "v11n": "YOLOv11n (Baseline)", "v11s": "YOLOv11s (Improved)",
        }
        logger.info(f"=== Step: Training {labels.get(variant, variant)} ===")
        return YOLOTrainer(self.config_path).train_variant(variant, dataset_yaml)

    # Backwards-compatible shortcuts
    def step_train_nano(self, dataset_yaml: str = "configs/dataset.yaml") -> str:
        return self.step_train_variant("v8n", dataset_yaml)

    def step_train_small(self, dataset_yaml: str = "configs/dataset.yaml") -> str:
        return self.step_train_variant("v8s", dataset_yaml)

    def step_evaluate(self, weights_path: str, model_name: str) -> Dict:
        logger.info(f"=== Step: Evaluating {model_name} ===")
        evaluator = ModelEvaluator(self.config_path)
        metrics = evaluator.evaluate(weights_path)
        self._save_json(metrics, f"reports/{model_name}_metrics.json")
        return metrics

    def step_compare_models(self, nano_metrics: Dict, small_metrics: Dict) -> Dict:
        logger.info("=== Step: Comparing Models ===")
        comparison = {
            "yolov8n": nano_metrics,
            "yolov8s": small_metrics,
            "recommendation": self._select_best_model(nano_metrics, small_metrics),
        }
        self._save_json(comparison, "reports/model_comparison.json")
        logger.info(f"Recommended model: {comparison['recommendation']}")
        return comparison

    def step_export_model(self, weights_path: str, format: str = "onnx") -> str:
        from ultralytics import YOLO
        logger.info(f"=== Step: Exporting Model to {format.upper()} ===")
        model = YOLO(weights_path)
        export_path = model.export(format=format)
        logger.info(f"Model exported: {export_path}")
        return str(export_path)

    def run_full_pipeline(
        self,
        dataset_yaml: str = "configs/dataset.yaml",
        train_nano: bool = True,
        train_small: bool = True,
        export_format: str = "onnx",
        yolo_version: str = "v8",          # "v8" or "v11"
    ) -> Dict:
        results = {}
        prefix = yolo_version              # "v8" or "v11"
        nano_key  = f"{prefix}n"
        small_key = f"{prefix}s"

        results["validation"] = self.step_validate_dataset()
        results["stats"] = self.step_generate_stats()

        nano_weights, small_weights = None, None

        if train_nano:
            nano_weights = self.step_train_variant(nano_key, dataset_yaml)
            results["nano_metrics"] = self.step_evaluate(nano_weights, f"yolo{nano_key}")

        if train_small:
            small_weights = self.step_train_variant(small_key, dataset_yaml)
            results["small_metrics"] = self.step_evaluate(small_weights, f"yolo{small_key}")

        if train_nano and train_small:
            results["comparison"] = self.step_compare_models(
                results["nano_metrics"], results["small_metrics"]
            )
            best_weights = (
                small_weights
                if results["comparison"]["recommendation"] == f"yolo{small_key}"
                else nano_weights
            )
        else:
            best_weights = small_weights or nano_weights

        if best_weights:
            results["export_path"] = self.step_export_model(best_weights, export_format)

        self._save_json(results, "reports/pipeline_summary.json")
        logger.info("=== Training Pipeline Complete ===")
        return results

    def _select_best_model(self, nano: Dict, small: Dict) -> str:
        nano_map = nano.get("map50", 0)
        small_map = small.get("map50", 0)
        return "yolov8s" if small_map >= nano_map else "yolov8n"

    def _save_json(self, data: Dict, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
