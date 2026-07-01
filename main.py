import argparse
import sys
from pathlib import Path

from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def cmd_train(args):
    from src.training.train_pipeline import TrainingPipeline
    pipeline = TrainingPipeline(args.config)

    if args.step == "all":
        pipeline.run_full_pipeline(
            dataset_yaml=args.dataset_yaml,
            train_nano=not args.skip_nano,
            train_small=not args.skip_small,
            export_format=args.export_format,
            yolo_version=args.yolo_version,
        )
    elif args.step == "validate":
        pipeline.step_validate_dataset()
    elif args.step == "stats":
        pipeline.step_generate_stats()
    elif args.step in ("nano", "small", "v11n", "v11s"):
        variant_map = {"nano": f"{args.yolo_version}n", "small": f"{args.yolo_version}s",
                       "v11n": "v11n", "v11s": "v11s"}
        pipeline.step_train_variant(variant_map[args.step], args.dataset_yaml)
    elif args.step == "export":
        if not args.weights:
            logger.error("--weights required for export step")
            sys.exit(1)
        pipeline.step_export_model(args.weights, args.export_format)
    else:
        logger.error(f"Unknown step: {args.step}")
        sys.exit(1)


def cmd_infer(args):
    from src.detection.inference import InferencePipeline
    pipeline = InferencePipeline(args.weights, args.config)

    if args.mode == "image":
        result = pipeline.run_image(args.source, save_output=True, output_dir=args.output_dir)
        logger.info(f"Vehicles: {result['counts']['total_vehicles']} | Density: {result['density']['density']}")

    elif args.mode == "folder":
        result = pipeline.run_folder(args.source, save_output=True, output_dir=args.output_dir)
        logger.info(f"Processed {result['summary']['total_images']} images")

    elif args.mode == "video":
        result = pipeline.run_video(
            args.source,
            output_path=args.output_video,
            max_frames=args.max_frames,
        )
        logger.info(f"Frames: {result['total_frames']} | Avg vehicles: {result['summary'].get('avg_vehicles', 0)}")

    elif args.mode == "webcam":
        result = pipeline.run_webcam(camera_index=args.camera, max_frames=args.max_frames)
        logger.info(f"Webcam session complete: {result['total_frames']} frames")

    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


def cmd_api(args):
    from src.api.app import run_server
    run_server(weights_path=args.weights, host=args.host, port=args.port)


def cmd_evaluate(args):
    from src.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(args.config)
    metrics = evaluator.evaluate(args.weights, args.dataset_yaml, split=args.split)
    logger.info(f"mAP@0.5: {metrics.get('map50', 0):.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics.get('map50_95', 0):.4f}")
    logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
    logger.info(f"Recall: {metrics.get('recall', 0):.4f}")


def cmd_test(args):
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="traffic-detect",
        description="Traffic Density Detection and Congestion Analysis",
    )
    parser.add_argument("--config", default="configs/settings.yaml")
    sub = parser.add_subparsers(dest="command")

    # Train
    p_train = sub.add_parser("train", help="Training pipeline")
    p_train.add_argument("--step", default="all",
                         choices=["all", "validate", "stats", "nano", "small", "v11n", "v11s", "export"])
    p_train.add_argument("--yolo-version", default="v8", choices=["v8", "v11"],
                         help="YOLO architecture version (default: v8)")
    p_train.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    p_train.add_argument("--weights", type=str, default="")
    p_train.add_argument("--skip-nano", action="store_true")
    p_train.add_argument("--skip-small", action="store_true")
    p_train.add_argument("--export-format", default="onnx", choices=["onnx", "torchscript", "tflite"])

    # Infer
    p_infer = sub.add_parser("infer", help="Run inference")
    p_infer.add_argument("--mode", required=True, choices=["image", "folder", "video", "webcam"])
    p_infer.add_argument("--source", type=str, default="")
    p_infer.add_argument("--weights", type=str, required=True)
    p_infer.add_argument("--output-dir", default="reports/inference")
    p_infer.add_argument("--output-video", type=str, default=None)
    p_infer.add_argument("--max-frames", type=int, default=None)
    p_infer.add_argument("--camera", type=int, default=0)

    # API
    p_api = sub.add_parser("api", help="Start FastAPI server")
    p_api.add_argument("--weights", type=str, default="")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)

    # Evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("--weights", type=str, required=True)
    p_eval.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])

    # Test
    sub.add_parser("test", help="Run unit tests")

    return parser


def main():
    setup_logger()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "train": cmd_train,
        "infer": cmd_infer,
        "api": cmd_api,
        "evaluate": cmd_evaluate,
        "test": cmd_test,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
