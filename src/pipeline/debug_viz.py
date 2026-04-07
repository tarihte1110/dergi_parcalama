from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.backends.ocr_base import OCRLine
from src.pipeline.models import TextBlock
from src.pipeline.visual_candidates import CandidateMetrics
from src.utils.image_ops import save_image


ROLE_COLOR = {
    "headline": (0, 140, 255),
    "content": (60, 220, 60),
    "other_text": (180, 180, 180),
}


def _draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]], color: tuple[int, int, int], label: str | None = None) -> np.ndarray:
    out = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(out, f"{label}{i}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def save_debug_views(
    debug_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    ocr_lines: list[OCRLine],
    text_blocks: list[TextBlock],
    text_mask: np.ndarray,
    non_text_mask: np.ndarray,
    visual_candidates: list[CandidateMetrics],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    ocr_overlay = _draw_boxes(image_bgr, [l.bbox_px for l in ocr_lines], (255, 255, 0), label="l")
    save_image(debug_dir / f"{stem}_01_ocr_lines.png", ocr_overlay)

    merged_overlay = _draw_boxes(image_bgr, [b.bbox_px for b in text_blocks], (255, 0, 0), label="t")
    save_image(debug_dir / f"{stem}_02_text_blocks.png", merged_overlay)

    role_overlay = image_bgr.copy()
    for i, block in enumerate(text_blocks, start=1):
        color = ROLE_COLOR.get(block.role, (200, 200, 200))
        x1, y1, x2, y2 = block.bbox_px
        cv2.rectangle(role_overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            role_overlay,
            f"{block.role}:{i}",
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    save_image(debug_dir / f"{stem}_03_roles.png", role_overlay)

    save_image(debug_dir / f"{stem}_04_text_mask.png", text_mask)
    save_image(debug_dir / f"{stem}_05_non_text_mask.png", non_text_mask)

    visual_overlay = _draw_boxes(image_bgr, [v.bbox for v in visual_candidates], (0, 0, 255), label="v")
    save_image(debug_dir / f"{stem}_06_visual_boxes.png", visual_overlay)
