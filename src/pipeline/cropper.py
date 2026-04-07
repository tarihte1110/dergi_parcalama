from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import CropConfig
from src.utils.geometry import BBox, clip_bbox
from src.utils.image_ops import save_image


def crop_with_padding(image_bgr: np.ndarray, bbox: BBox, cfg: CropConfig) -> tuple[np.ndarray, BBox]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(cfg.crop_padding_px, int(round(min(h, w) * cfg.crop_padding_ratio)))
    padded = clip_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h)
    px1, py1, px2, py2 = padded
    return image_bgr[py1:py2, px1:px2].copy(), padded


def save_visual_crop(image_bgr: np.ndarray, bbox: BBox, cfg: CropConfig, out_path: Path) -> BBox:
    crop, padded_bbox = crop_with_padding(image_bgr, bbox, cfg)
    save_image(out_path, crop)
    return padded_bbox
