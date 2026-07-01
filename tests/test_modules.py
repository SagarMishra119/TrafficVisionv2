import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestVehicleCounter(unittest.TestCase):
    def setUp(self):
        from src.counting.counter import VehicleCounter
        self.counter = VehicleCounter()

    def test_empty_detections(self):
        result = self.counter.count([])
        self.assertEqual(result["total_vehicles"], 0)
        self.assertEqual(result["cars"], 0)

    def test_count_mixed_vehicles(self):
        detections = [
            {"vehicle_type": "car", "confidence": 0.9, "bbox": [], "class_id": 0},
            {"vehicle_type": "car", "confidence": 0.8, "bbox": [], "class_id": 0},
            {"vehicle_type": "bus", "confidence": 0.7, "bbox": [], "class_id": 1},
            {"vehicle_type": "truck", "confidence": 0.85, "bbox": [], "class_id": 2},
        ]
        result = self.counter.count(detections)
        self.assertEqual(result["total_vehicles"], 4)
        self.assertEqual(result["cars"], 2)
        self.assertEqual(result["buses"], 1)
        self.assertEqual(result["trucks"], 1)

    def test_count_by_confidence(self):
        detections = [
            {"vehicle_type": "car", "confidence": 0.9, "bbox": [], "class_id": 0},
            {"vehicle_type": "car", "confidence": 0.3, "bbox": [], "class_id": 0},
        ]
        result = self.counter.count_by_confidence(detections, min_confidence=0.5)
        self.assertEqual(result["total_vehicles"], 1)

    def test_aggregate_video_counts(self):
        frames = [
            {"total_vehicles": 10, "cars": 8, "buses": 1, "trucks": 1, "motorcycles": 0, "bicycles": 0},
            {"total_vehicles": 20, "cars": 15, "buses": 3, "trucks": 2, "motorcycles": 0, "bicycles": 0},
        ]
        result = self.counter.aggregate_video_counts(frames)
        self.assertEqual(result["peak"]["total_vehicles"], 20)
        self.assertEqual(result["average"]["total_vehicles"], 15.0)


class TestDensityClassifier(unittest.TestCase):
    def setUp(self):
        from src.density.classifier import DensityClassifier
        self.classifier = DensityClassifier()

    def test_low_density(self):
        result = self.classifier.classify(5)
        self.assertEqual(result["density"], "Low")
        self.assertEqual(result["vehicle_count"], 5)

    def test_medium_density(self):
        result = self.classifier.classify(15)
        self.assertEqual(result["density"], "Medium")

    def test_high_density(self):
        result = self.classifier.classify(30)
        self.assertEqual(result["density"], "High")

    def test_boundary_low_max(self):
        result = self.classifier.classify(10)
        self.assertEqual(result["density"], "Low")

    def test_boundary_medium_min(self):
        result = self.classifier.classify(11)
        self.assertEqual(result["density"], "Medium")

    def test_zero_vehicles(self):
        result = self.classifier.classify(0)
        self.assertEqual(result["density"], "Low")

    def test_threshold_update(self):
        self.classifier.update_thresholds(low_max=5, medium_max=15)
        self.assertEqual(self.classifier.classify(3)["density"], "Low")
        self.assertEqual(self.classifier.classify(10)["density"], "Medium")
        self.assertEqual(self.classifier.classify(20)["density"], "High")

    def test_density_trend_increasing(self):
        history = [{"vehicle_count": 5}, {"vehicle_count": 15}]
        trend = self.classifier.get_density_trend(history)
        self.assertEqual(trend["trend"], "increasing")

    def test_density_trend_stable(self):
        history = [{"vehicle_count": 10}, {"vehicle_count": 11}]
        trend = self.classifier.get_density_trend(history)
        self.assertEqual(trend["trend"], "stable")


class TestTrafficAnalytics(unittest.TestCase):
    def setUp(self):
        from src.analytics.analytics import TrafficAnalytics
        self.analytics = TrafficAnalytics(reports_dir="/tmp/test_reports")

    def tearDown(self):
        self.analytics.clear_session()

    def _make_counts(self, total=5):
        return {"total_vehicles": total, "cars": total, "buses": 0, "trucks": 0, "motorcycles": 0, "bicycles": 0}

    def test_log_and_summary(self):
        for i in range(10):
            self.analytics.log_frame(i, [], self._make_counts(i + 1), {"density": "Low"})
        summary = self.analytics.get_session_summary()
        self.assertEqual(summary["total_frames"], 10)
        self.assertEqual(summary["max_vehicles"], 10)

    def test_empty_summary(self):
        summary = self.analytics.get_session_summary()
        self.assertEqual(summary, {})

    def test_congestion_index_zero(self):
        idx = self.analytics.compute_congestion_index(self._make_counts(0))
        self.assertEqual(idx, 0.0)

    def test_congestion_index_max(self):
        large_counts = {"cars": 100, "buses": 50, "trucks": 50, "motorcycles": 50, "bicycles": 50}
        idx = self.analytics.compute_congestion_index(large_counts)
        self.assertEqual(idx, 100.0)

    def test_realtime_stats(self):
        for i in range(15):
            self.analytics.log_frame(i, [], self._make_counts(i + 1), {"density": "Low"})
        stats = self.analytics.get_realtime_stats(window=5)
        self.assertEqual(stats["window_frames"], 5)
        self.assertIn("current_density", stats)


class TestDatasetValidator(unittest.TestCase):
    def test_empty_directory(self):
        from src.data.dataset_validator import DatasetValidator
        validator = DatasetValidator("/nonexistent/path")
        report = validator.validate_split("train")
        self.assertEqual(report["total_images"], 0)

    def test_valid_label_format(self):
        import tempfile, os
        from src.data.dataset_validator import DatasetValidator
        validator = DatasetValidator("/tmp")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("1 0.3 0.4 0.1 0.2\n")
            tmp = f.name
        errors = validator.validate_label_format(tmp, num_classes=5)
        self.assertEqual(errors, [])
        os.unlink(tmp)

    def test_invalid_label_class(self):
        import tempfile, os
        from src.data.dataset_validator import DatasetValidator
        validator = DatasetValidator("/tmp")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("99 0.5 0.5 0.2 0.3\n")
            tmp = f.name
        errors = validator.validate_label_format(tmp, num_classes=5)
        self.assertTrue(len(errors) > 0)
        os.unlink(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
