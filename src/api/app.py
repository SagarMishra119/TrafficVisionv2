import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router, set_detector
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

_weights_path: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    if _weights_path and Path(_weights_path).exists():
        from src.detection.detector import VehicleDetector
        detector = VehicleDetector(_weights_path)
        set_detector(detector)
        logger.info(f"Model ready: {_weights_path}")
    else:
        logger.warning("No weights loaded. Prediction endpoints will return 503.")
    yield
    logger.info("Shutting down server")


def create_app(weights_path: str = "") -> FastAPI:
    global _weights_path
    _weights_path = weights_path

    config = load_config()
    app = FastAPI(
        title="Traffic Density Detection API",
        description="Vehicle detection, counting, and traffic density analysis using YOLOv8.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["Traffic Detection"])

    # Serve reports folder
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

    # Serve the UI
    app_dir = Path("app")
    if app_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(app_dir)), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        index = Path("app/index.html")
        if index.exists():
            return FileResponse(str(index))
        return {
            "service": "Traffic Density Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


def run_server(weights_path: str = "", host: str = "0.0.0.0", port: int = 8000):
    app = create_app(weights_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="Path to YOLO weights")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.weights, args.host, args.port)
