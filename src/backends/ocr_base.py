from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.utils.geometry import BBox


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    bbox_px: BBox
    line_height: float
    line_width: float
    center: tuple[float, float]


class OCRBackend(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[OCRLine]:
        """Detect text lines and return normalized OCRLine objects."""

    def detect_batch(self, images: list[np.ndarray]) -> list[list[OCRLine]]:
        """Default batch implementation; backends can override with vectorized inference."""
        return [self.detect(img) for img in images]
