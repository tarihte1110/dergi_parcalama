from __future__ import annotations

import cv2
import numpy as np

from src.config import MaskConfig
from src.utils.image_ops import ratio_to_kernel


def build_non_text_mask(image_bgr: np.ndarray, text_mask: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # İyileştirilmiş beyaz arka plan tespiti - açık renkli çizimleri koru
    sat = hsv[:, :, 1]

    # Adaptive local thresholding - global yerine yerel ortalama kullan
    white_local_mean = cv2.GaussianBlur(gray.astype(np.float32), (51, 51), 0)
    is_white_bg = (
        (gray >= cfg.white_threshold) &
        (sat <= cfg.low_saturation_threshold) &
        (np.abs(gray.astype(np.float32) - white_local_mean) < 8)
    )

    foreground = (~is_white_bg).astype(np.uint8) * 255

    # Texture ve color analizi ile görsel bölgeleri güçlendir
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    color_std = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), 15.0)
    color_var = np.std(color_std, axis=2)
    colorful_regions = color_var > 12.0

    # Edge-based texture detection
    edges = cv2.Canny(gray, 80, 160) > 0

    # Local std ile texture detection
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 3.0)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)
    textured = std > cfg.visual_texture_std_threshold

    # Tüm görsel ipuçlarını birleştir
    visual_hints = (textured | colorful_regions | edges).astype(np.uint8) * 255

    # Foreground VEYA visual hints (birleşim)
    non_text = cv2.bitwise_or(foreground, visual_hints)

    # Text mask'i çıkar
    non_text = cv2.bitwise_and(non_text, cv2.bitwise_not(text_mask))

    # Morphological operations'ı daha hafif yap - detay kaybını azalt
    open_k = ratio_to_kernel((h, w), cfg.non_text_open_ratio * 0.7, min_size=1)  # %30 azalt
    close_k = ratio_to_kernel((h, w), cfg.non_text_close_ratio * 0.8, min_size=1)  # %20 azalt
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    non_text = cv2.morphologyEx(non_text, cv2.MORPH_OPEN, open_kernel, iterations=1)
    non_text = cv2.morphologyEx(non_text, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return non_text
