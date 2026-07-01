import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from src.analytics.analytics import TrafficAnalytics
from src.analytics.visualizer import TrafficVisualizer
from src.counting.counter import VehicleCounter
from src.density.classifier import DensityClassifier
from src.detection.detector import VehicleDetector
from src.utils.config import load_config
from src.utils.image_utils import save_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    def __init__(
        self,
        weights_path: str,
        config_path: str = "configs/settings.yaml",
    ):
        self.config = load_config(config_path)
        self.detector = VehicleDetector(weights_path, config_path)
        self.counter = VehicleCounter(config_path)
        self.classifier = DensityClassifier(config_path)
        self.analytics = TrafficAnalytics()
        self.visualizer = TrafficVisualizer()

    def run_image(
        self,
        image_path: str,
        save_output: bool = True,
        output_dir: str = "reports/inference",
    ) -> Dict:
        detections, annotated = self.detector.detect_from_path(image_path)
        counts = self.counter.count(detections)
        density = self.classifier.classify(counts["total_vehicles"])
        congestion = self.analytics.compute_congestion_index(counts)

        output_path = None
        if save_output:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            output_path = str(Path(output_dir) / f"{stem}_detected.jpg")
            save_image(annotated, output_path)

        result = {
            "source": image_path,
            "mode": "image",
            "detections": detections,
            "counts": counts,
            "density": density,
            "congestion_index": congestion,
            "output_image": output_path,
        }
        self._save_result(result, output_dir, Path(image_path).stem)
        return result

    def run_folder(
        self,
        folder_path: str,
        save_output: bool = True,
        output_dir: str = "reports/inference",
    ) -> Dict:
        from src.utils.file_utils import get_image_paths
        images = get_image_paths(folder_path)
        results = {}

        for img_path in images:
            try:
                result = self.run_image(str(img_path), save_output, output_dir)
                results[img_path.name] = result
            except Exception as e:
                logger.error(f"Error on {img_path.name}: {e}")

        summary = {
            "total_images": len(results),
            "total_detections": sum(r["counts"]["total_vehicles"] for r in results.values()),
            "density_breakdown": {},
        }
        for r in results.values():
            d = r["density"]["density"]
            summary["density_breakdown"][d] = summary["density_breakdown"].get(d, 0) + 1

        logger.info(f"Folder inference complete: {len(results)} images")
        return {"results": results, "summary": summary}

    def run_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict:
        self.analytics.clear_session()
        frame_id = 0

        for fid, detections, annotated in self.detector.detect_video(video_path, output_path):
            if max_frames and frame_id >= max_frames:
                break
            counts = self.counter.count(detections)
            density = self.classifier.classify(counts["total_vehicles"])
            self.analytics.log_frame(fid, detections, counts, density)
            frame_id += 1
            if frame_id % 50 == 0:
                logger.info(f"Processed {frame_id} frames...")

        summary = self.analytics.get_session_summary()
        timeline_plot = self.visualizer.plot_vehicle_timeline(self.analytics._session_log)
        density_plot = self.visualizer.plot_density_distribution(self.analytics._session_log)

        result = {
            "source": video_path,
            "mode": "video",
            "total_frames": frame_id,
            "output_video": output_path,
            "summary": summary,
            "plots": {"timeline": timeline_plot, "density": density_plot},
        }
        self._save_result(result, "reports", "video_inference")
        self.analytics.export_to_csv("video_analytics.csv")
        return result

    def run_webcam(self, camera_index: int = 0, max_frames: Optional[int] = None) -> Dict:
        self.analytics.clear_session()
        frame_id = 0
        logger.info("Starting webcam inference. Press Ctrl+C to stop.")

        try:
            for fid, detections, annotated in self.detector.detect_webcam(camera_index):
                if max_frames and frame_id >= max_frames:
                    break
                counts = self.counter.count(detections)
                density = self.classifier.classify(counts["total_vehicles"])
                self.analytics.log_frame(fid, detections, counts, density)

                cv2.imshow("Traffic Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_id += 1
        except KeyboardInterrupt:
            logger.info("Webcam stopped by user")
        finally:
            cv2.destroyAllWindows()

        return {
            "source": f"webcam:{camera_index}",
            "mode": "webcam",
            "total_frames": frame_id,
            "summary": self.analytics.get_session_summary(),
        }

    def _save_result(self, result: Dict, output_dir: str, name: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output = Path(output_dir) / f"{name}_result.json"
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
