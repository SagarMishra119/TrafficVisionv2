from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.image_utils import draw_bounding_box
from src.utils.logger import get_logger

logger = get_logger(__name__)

VEHICLE_COLORS = {
    "car": (0, 255, 0),
    "bus": (255, 165, 0),
    "truck": (255, 0, 0),
    "motorcycle": (0, 0, 255),
    "bicycle": (255, 0, 255),
}


class VehicleDetector:
    def __init__(
        self,
        weights_path: str,
        config_path: str = "configs/settings.yaml",
    ):
        self.config = load_config(config_path)
        self.classes = self.config["classes"]
        self.inference_cfg = self.config["inference"]
        self.model = YOLO(weights_path)
        logger.info(f"Model loaded: {weights_path}")

    def detect_image(self, image: np.ndarray) -> List[Dict]:
        results = self.model.predict(
            source=image,
            conf=self.inference_cfg["confidence"],
            iou=self.inference_cfg["iou_threshold"],
            max_det=self.inference_cfg["max_detections"],
            verbose=False,
        )
        return self._parse_results(results[0])

    def detect_from_path(self, image_path: str) -> Tuple[List[Dict], np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        detections = self.detect_image(image)
        annotated = self.annotate_image(image.copy(), detections)
        return detections, annotated

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> Generator[Tuple[int, List[Dict], np.ndarray], None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.detect_image(frame)
                annotated = self.annotate_image(frame.copy(), detections)
                if writer:
                    writer.write(annotated)
                yield frame_id, detections, annotated
                frame_id += 1
        finally:
            cap.release()
            if writer:
                writer.release()

    def detect_webcam(self, camera_index: int = 0) -> Generator[Tuple[int, List[Dict], np.ndarray], None, None]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam: {camera_index}")
        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.detect_image(frame)
                annotated = self.annotate_image(frame.copy(), detections)
                yield frame_id, detections, annotated
                frame_id += 1
        finally:
            cap.release()

    def detect_folder(self, folder_path: str) -> Dict[str, List[Dict]]:
        from src.utils.file_utils import get_image_paths
        images = get_image_paths(folder_path)
        all_results = {}
        for img_path in images:
            try:
                detections, _ = self.detect_from_path(str(img_path))
                all_results[img_path.name] = detections
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
        logger.info(f"Processed {len(all_results)} images")
        return all_results

    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            color = VEHICLE_COLORS.get(det["vehicle_type"], (128, 128, 128))
            image = draw_bounding_box(
                image,
                det["bbox"],
                det["vehicle_type"],
                det["confidence"],
                color=color,
            )
        return image

    def _parse_results(self, result) -> List[Dict]:
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls_id = int(box.cls.item())
            if cls_id >= len(self.classes):
                continue
            detections.append({
                "vehicle_type": self.classes[cls_id],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
                "class_id": cls_id,
            })
        return detections
