
import shutil
from pathlib import Path
from typing import Dict, Literal, Optional

from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ModelVariant = Literal["v8n", "v8s", "v11n", "v11s"]

_MODEL_MAP = {
    "v8n":  ("model_nano",      "yolov8n_traffic"),
    "v8s":  ("model_small",     "yolov8s_traffic"),
    "v11n": ("model_v11_nano",  "yolo11n_traffic"),
    "v11s": ("model_v11_small", "yolo11s_traffic"),
}


class YOLOTrainer:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.training_cfg = self.config["training"]
        self.models_dir = Path(self.config["paths"]["models"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        model_name: str,
        dataset_yaml: str = "configs/dataset.yaml",
        run_name: Optional[str] = None,
        override: Optional[Dict] = None,
    ) -> str:
        cfg = {**self.training_cfg, **(override or {})}
        run_name = run_name or model_name.replace(".pt", "")
        logger.info(f"Starting training: {model_name} | Run: {run_name}")

        model = YOLO(model_name)
        model.train(
            data=dataset_yaml,
            epochs=cfg["epochs"],
            imgsz=cfg["image_size"],
            batch=cfg["batch_size"],
            lr0=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            momentum=cfg["momentum"],
            patience=cfg["patience"],
            workers=cfg["workers"],
            device=cfg.get("device", "cpu"),
            optimizer=cfg["optimizer"],
            cos_lr=cfg.get("cos_lr", True),
            amp=cfg.get("amp", True),
            cache=cfg.get("cache", False),
            name=run_name,
            project="models/runs",
            exist_ok=True,
            verbose=True,
        )

        saved = self.models_dir / f"{run_name}_best.pt"
        best = self._find_best_weights(run_name)
        if best:
            shutil.copy2(best, saved)
            logger.info(f"Best weights saved: {saved}")
        else:
            logger.warning(f"best.pt not found for '{run_name}'")
        return str(saved)

    def _find_best_weights(self, run_name: str) -> Optional[Path]:
        """Search common YOLO output locations for best.pt."""
        candidates = [
            Path(f"models/runs/{run_name}/weights/best.pt"),
            Path(f"runs/detect/models/runs/{run_name}/weights/best.pt"),
            Path(f"runs/detect/{run_name}/weights/best.pt"),
        ]
        # Also glob in case Ultralytics nests further
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found weights at: {candidate}")
                return candidate

        # Fallback: recursive search from cwd
        matches = list(Path(".").rglob(f"{run_name}/weights/best.pt"))
        if matches:
            logger.info(f"Found weights via search: {matches[0]}")
            return matches[0]

        return None

    def train_variant(
        self,
        variant: ModelVariant = "v8s",
        dataset_yaml: str = "configs/dataset.yaml",
        override: Optional[Dict] = None,
    ) -> str:
        cfg_key, run_name = _MODEL_MAP[variant]
        model_name = self.training_cfg[cfg_key]
        return self.train(model_name, dataset_yaml, run_name, override)

    def train_nano(self, dataset_yaml: str = "configs/dataset.yaml") -> str:
        return self.train_variant("v8n", dataset_yaml)

    def train_small(self, dataset_yaml: str = "configs/dataset.yaml") -> str:
        return self.train_variant("v8s", dataset_yaml)

    def train_v11_nano(self, dataset_yaml: str = "configs/dataset.yaml") -> str:
        return self.train_variant("v11n", dataset_yaml)

    def train_v11_small(self, dataset_yaml: str = "configs/dataset.yaml") -> str:
        return self.train_variant("v11s", dataset_yaml)

    def resume_training(self, run_name: str) -> None:
        last = self._find_best_weights(run_name)
        if not last:
            raise FileNotFoundError(f"No checkpoint found for '{run_name}'")
        YOLO(str(last)).train(resume=True)
        logger.info(f"Resumed: {last}")