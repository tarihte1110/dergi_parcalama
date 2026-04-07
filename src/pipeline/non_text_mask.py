from __future__ import annotations

import cv2
import numpy as np

from src.config import MaskConfig
from src.utils.image_ops import ratio_to_kernel


def build_non_text_mask(image_bgr: np.ndarray, text_mask: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    sat = hsv[:, :, 1]
    is_white_bg = (gray >= cfg.white_threshold) & (sat <= cfg.low_saturation_threshold)
    foreground = (~is_white_bg).astype(np.uint8) * 255

    non_text = cv2.bitwise_and(foreground, cv2.bitwise_not(text_mask))

    open_k = ratio_to_kernel((h, w), cfg.non_text_open_ratio, min_size=1)
    close_k = ratio_to_kernel((h, w), cfg.non_text_close_ratio, min_size=1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    non_text = cv2.morphologyEx(non_text, cv2.MORPH_OPEN, open_kernel, iterations=1)
    non_text = cv2.morphologyEx(non_text, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return non_text
