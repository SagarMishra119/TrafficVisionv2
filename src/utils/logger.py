import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_dir: str = "reports/logs", level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(
        f"{log_dir}/traffic_detection.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )


def get_logger(name: str):
    return logger.bind(name=name)
