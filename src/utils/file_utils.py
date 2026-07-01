import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_paths(directory: str, recursive: bool = False) -> List[Path]:
    base = Path(directory)
    pattern = "**/*" if recursive else "*"
    return [
        p for p in base.glob(pattern)
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def get_label_paths(directory: str) -> List[Path]:
    return list(Path(directory).glob("*.txt"))


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def remove_file(path: str) -> None:
    if Path(path).exists():
        os.remove(path)


def count_files(directory: str, extension: Optional[str] = None) -> int:
    base = Path(directory)
    if not base.exists():
        return 0
    if extension:
        return len(list(base.glob(f"*{extension}")))
    return len([f for f in base.iterdir() if f.is_file()])
