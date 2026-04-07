from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower().lstrip(".") or "png"
    success, encoded = cv2.imencode(f".{suffix}", image)
    if not success:
        raise ValueError(f"Failed to encode image for saving: {path}")
    encoded.tofile(str(path))


def ensure_odd_kernel(size: int) -> int:
    size = max(1, size)
    if size % 2 == 0:
        size += 1
    return size


def ratio_to_kernel(image_shape: tuple[int, int], ratio: float, min_size: int = 1) -> int:
    h, w = image_shape[:2]
    base = int(round(min(h, w) * ratio))
    return ensure_odd_kernel(max(min_size, base))
