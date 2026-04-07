from __future__ import annotations

import cv2
import numpy as np

from src.config import MaskConfig
from src.pipeline.models import TextBlock
from src.utils.image_ops import ratio_to_kernel


def build_text_mask(image_shape: tuple[int, int], blocks: list[TextBlock], cfg: MaskConfig) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for block in blocks:
        for x1, y1, x2, y2 in block.regions_px:
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)

    kernel_size = ratio_to_kernel((h, w), cfg.text_dilation_ratio, min_size=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated
