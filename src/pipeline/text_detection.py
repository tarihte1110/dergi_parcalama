from __future__ import annotations

import numpy as np
import cv2

from src.backends.ocr_base import OCRBackend, OCRLine
from src.config import OCRConfig
from src.utils.geometry import iou


def detect_text_lines(image: np.ndarray, backend: OCRBackend) -> list[OCRLine]:
    return backend.detect(image)


def _enhance_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
    sharp = cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
    return sharp


def _merge_line_sets(primary: list[OCRLine], secondary: list[OCRLine], merge_iou: float) -> list[OCRLine]:
    out = list(primary)
    for cand in secondary:
        replaced = False
        for i, cur in enumerate(out):
            if iou(cand.bbox_px, cur.bbox_px) >= merge_iou:
                if cand.confidence > cur.confidence:
                    out[i] = cand
                replaced = True
                break
        if not replaced:
            out.append(cand)
    out.sort(key=lambda l: (l.bbox_px[1], l.bbox_px[0]))
    return out


def detect_text_lines_adaptive(image: np.ndarray, backend: OCRBackend, cfg: OCRConfig) -> list[OCRLine]:
    lines = backend.detect(image)
    if not cfg.enable_second_pass:
        return lines
    if not lines:
        enhanced = _enhance_for_ocr(image)
        return backend.detect(enhanced)

    avg_conf = float(sum(l.confidence for l in lines) / max(1, len(lines)))
    if (avg_conf >= cfg.second_pass_min_avg_conf) and (len(lines) >= cfg.second_pass_min_lines):
        return lines

    enhanced = _enhance_for_ocr(image)
    second = backend.detect(enhanced)
    if not second:
        return lines
    return _merge_line_sets(lines, second, merge_iou=cfg.second_pass_merge_iou)
