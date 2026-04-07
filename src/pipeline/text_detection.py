from __future__ import annotations

import numpy as np

from src.backends.ocr_base import OCRBackend, OCRLine


def detect_text_lines(image: np.ndarray, backend: OCRBackend) -> list[OCRLine]:
    return backend.detect(image)
