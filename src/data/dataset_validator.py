import cv2
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from src.utils.file_utils import file_md5, get_image_paths
from src.utils.image_utils import is_valid_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def validate_split(self, split: str) -> Dict:
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"

        report = {
            "split": split,
            "total_images": 0,
            "corrupt_images": [],
            "missing_labels": [],
            "empty_labels": [],
            "duplicate_images": [],
            "valid_pairs": 0,
        }

        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            return report

        images = get_image_paths(str(img_dir))
        report["total_images"] = len(images)

        md5_map = defaultdict(list)
        for img_path in images:
            if not is_valid_image(str(img_path)):
                report["corrupt_images"].append(str(img_path))
                continue

            md5 = file_md5(str(img_path))
            md5_map[md5].append(str(img_path))

            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                report["missing_labels"].append(str(img_path))
            elif label_path.stat().st_size == 0:
                report["empty_labels"].append(str(img_path))
            else:
                report["valid_pairs"] += 1

        for md5, paths in md5_map.items():
            if len(paths) > 1:
                report["duplicate_images"].extend(paths[1:])

        return report

    def validate_all(self) -> Dict:
        full_report = {}
        for split in ["train", "valid", "test"]:
            full_report[split] = self.validate_split(split)
            self._log_split_report(full_report[split])
        return full_report

    def validate_label_format(self, label_path: str, num_classes: int) -> List[str]:
        errors = []
        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"Line {i+1}: expected 5 values, got {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"Line {i+1}: non-numeric values")
                    continue
                if cls_id < 0 or cls_id >= num_classes:
                    errors.append(f"Line {i+1}: invalid class id {cls_id}")
                if not all(0.0 <= v <= 1.0 for v in coords):
                    errors.append(f"Line {i+1}: coordinates out of [0,1] range")
        return errors

    def _log_split_report(self, report: Dict) -> None:
        split = report["split"]
        logger.info(f"[{split}] Total: {report['total_images']} | Valid pairs: {report['valid_pairs']}")
        if report["corrupt_images"]:
            logger.warning(f"[{split}] Corrupt images: {len(report['corrupt_images'])}")
        if report["missing_labels"]:
            logger.warning(f"[{split}] Missing labels: {len(report['missing_labels'])}")
        if report["empty_labels"]:
            logger.warning(f"[{split}] Empty labels: {len(report['empty_labels'])}")
        if report["duplicate_images"]:
            logger.warning(f"[{split}] Duplicates: {len(report['duplicate_images'])}")
