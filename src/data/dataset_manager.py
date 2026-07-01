import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.config import load_config
from src.utils.file_utils import get_image_paths, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetManager:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.dataset_path = Path(self.config["paths"]["dataset"])
        self.splits = {
            "train": self.config["dataset"]["train_split"],
            "valid": self.config["dataset"]["valid_split"],
            "test": self.config["dataset"]["test_split"],
        }

    def download_dataset_roboflow(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        output_format: str = "yolov8",
    ) -> str:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(output_format)
        logger.info(f"Dataset downloaded to: {dataset.location}")
        return dataset.location

    def organize_from_flat(self, images_dir: str, labels_dir: str) -> None:
        images = get_image_paths(images_dir)
        random.seed(42)
        random.shuffle(images)

        n = len(images)
        train_end = int(n * self.splits["train"])
        valid_end = train_end + int(n * self.splits["valid"])

        split_indices = {
            "train": images[:train_end],
            "valid": images[train_end:valid_end],
            "test": images[valid_end:],
        }

        for split, img_list in split_indices.items():
            img_out = ensure_dir(self.dataset_path / split / "images")
            lbl_out = ensure_dir(self.dataset_path / split / "labels")

            for img_path in img_list:
                label_path = Path(labels_dir) / (img_path.stem + ".txt")
                shutil.copy2(img_path, img_out / img_path.name)
                if label_path.exists():
                    shutil.copy2(label_path, lbl_out / label_path.name)

            logger.info(f"[{split}] {len(img_list)} images organized")

    def get_split_counts(self) -> Dict[str, int]:
        counts = {}
        for split in ["train", "valid", "test"]:
            img_dir = self.dataset_path / split / "images"
            counts[split] = len(get_image_paths(str(img_dir))) if img_dir.exists() else 0
        return counts

    def generate_yaml(self, output_path: str = "configs/dataset.yaml") -> None:
        classes = self.config["classes"]
        content = {
            "path": str(self.dataset_path.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(classes),
            "names": classes,
        }
        import yaml
        with open(output_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False)
        logger.info(f"Dataset YAML generated: {output_path}")

    def get_all_image_label_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        pairs = []
        for img in get_image_paths(str(img_dir)):
            lbl = lbl_dir / (img.stem + ".txt")
            pairs.append((img, lbl))
        return pairs
