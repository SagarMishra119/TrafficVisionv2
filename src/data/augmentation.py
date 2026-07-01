import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_augmentation_pipeline(config: Dict) -> A.Compose:
    aug_cfg = config.get("augmentation", {})
    return A.Compose(
        [
            A.HorizontalFlip(p=aug_cfg.get("fliplr", 0.5)),
            A.Rotate(limit=aug_cfg.get("degrees", 10), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("hsv_v", 0.4),
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(aug_cfg.get("hsv_h", 0.015) * 180),
                sat_shift_limit=int(aug_cfg.get("hsv_s", 0.7) * 255),
                val_shift_limit=int(aug_cfg.get("hsv_v", 0.4) * 255),
                p=0.4,
            ),
            A.RandomScale(scale_limit=aug_cfg.get("scale", 0.5) * 0.5, p=0.3),
            A.GaussNoise(p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def parse_yolo_label(label_path: str) -> Tuple[List[int], List[List[float]]]:
    class_labels, bboxes = [], []
    if not Path(label_path).exists():
        return class_labels, bboxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(v) for v in parts[1:]])
    return class_labels, bboxes


def save_yolo_label(
    output_path: str,
    class_labels: List[int],
    bboxes: List[List[float]],
) -> None:
    with open(output_path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")


class DataAugmentor:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = load_config(config_path)
        self.pipeline = build_augmentation_pipeline(self.config)

    def augment_image_label(
        self,
        image: np.ndarray,
        class_labels: List[int],
        bboxes: List[List[float]],
    ) -> Tuple[np.ndarray, List[int], List[List[float]]]:
        if not bboxes:
            result = self.pipeline(image=image, bboxes=[], class_labels=[])
            return result["image"], [], []

        result = self.pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        return result["image"], result["class_labels"], result["bboxes"]

    def augment_dataset_split(
        self,
        img_dir: str,
        lbl_dir: str,
        out_img_dir: str,
        out_lbl_dir: str,
        num_augmentations: int = 2,
    ) -> None:
        from src.utils.file_utils import get_image_paths, ensure_dir
        import cv2

        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        images = get_image_paths(img_dir)
        logger.info(f"Augmenting {len(images)} images x{num_augmentations}")

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
            class_labels, bboxes = parse_yolo_label(str(lbl_path))

            for i in range(num_augmentations):
                aug_img, aug_cls, aug_bboxes = self.augment_image_label(
                    image.copy(), class_labels.copy(), [b.copy() for b in bboxes]
                )
                out_name = f"{img_path.stem}_aug{i}"
                cv2.imwrite(str(Path(out_img_dir) / f"{out_name}.jpg"), aug_img)
                save_yolo_label(
                    str(Path(out_lbl_dir) / f"{out_name}.txt"),
                    aug_cls,
                    aug_bboxes,
                )

        logger.info("Augmentation complete")
