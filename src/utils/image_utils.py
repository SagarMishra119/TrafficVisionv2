import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    return img


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def is_valid_image(path: str) -> bool:
    try:
        img = cv2.imread(path)
        return img is not None and img.size > 0
    except Exception:
        return False


def draw_bounding_box(
    image: np.ndarray,
    bbox: list,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {confidence:.2f}"
    font_scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return image


def encode_image_base64(image: np.ndarray) -> str:
    import base64
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def save_image(image: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)
