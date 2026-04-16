from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from src.backends.ocr_base import OCRLine


PageLayoutType = Literal[
    "background_photo_page",
    "comic_panel_page",
    "puzzle_page",
    "activity_page",
    "framed_photo_page",
    "mixed_page",
]


@dataclass(frozen=True)
class PageLayoutDecision:
    page_type: PageLayoutType
    confidence: float
    text_ratio: float
    non_text_ratio: float
    edge_density: float
    rect_count: int
    grid_score: float


def _count_rectangles(gray: np.ndarray) -> int:
    edge = cv2.Canny(gray, 70, 150)
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if (w * h) >= 22000:
                cnt += 1
    return cnt


def _grid_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ax = np.abs(gx)
    ay = np.abs(gy)
    x_peaky = float(np.percentile(ax, 92))
    y_peaky = float(np.percentile(ay, 92))
    x_mean = float(np.mean(ax))
    y_mean = float(np.mean(ay))
    # Higher when many strong horizontal+vertical straight edges exist.
    score = ((x_peaky / max(1e-3, x_mean)) + (y_peaky / max(1e-3, y_mean))) / 2.0
    return float(score)


def classify_page_layout(
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    non_text_mask: np.ndarray,
    ocr_lines: list[OCRLine],
) -> PageLayoutDecision:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges > 0)) / float(max(1, edges.size))
    text_ratio = float(np.count_nonzero(text_mask > 0)) / float(max(1, text_mask.size))
    non_text_ratio = float(np.count_nonzero(non_text_mask > 0)) / float(max(1, non_text_mask.size))
    rect_count = _count_rectangles(gray)
    grid_score = _grid_score(gray)
    joined_text = " ".join((ln.text or "") for ln in ocr_lines).lower()
    puzzle_keywords = (
        "bulmaca",
        "kelime av",
        "kelime avi",
        "çengel",
        "sudoku",
        "soldan sağa",
        "yukarıdan aşağıya",
        "şifre",
        "rakamli",
        "rakamlı",
    )
    puzzle_kw_hits = sum(1 for kw in puzzle_keywords if kw in joined_text)

    # Strong background-photo signal: visual-heavy, low-to-mid text, low rectangular structure.
    if non_text_ratio >= 0.72 and text_ratio <= 0.32 and rect_count <= 2 and edge_density >= 0.02:
        return PageLayoutDecision(
            page_type="background_photo_page",
            confidence=0.80,
            text_ratio=text_ratio,
            non_text_ratio=non_text_ratio,
            edge_density=edge_density,
            rect_count=rect_count,
            grid_score=grid_score,
        )

    # Puzzle pages often show strong grid-like orthogonal edges and mid text.
    if (grid_score >= 5.4 and text_ratio >= 0.08) or (puzzle_kw_hits >= 2 and text_ratio >= 0.08):
        return PageLayoutDecision(
            page_type="puzzle_page",
            confidence=0.82 if puzzle_kw_hits >= 2 else 0.74,
            text_ratio=text_ratio,
            non_text_ratio=non_text_ratio,
            edge_density=edge_density,
            rect_count=rect_count,
            grid_score=grid_score,
        )

    # Comic/activity-like pages typically have many OCR lines and visible panel boundaries.
    if rect_count >= 3 and len(ocr_lines) >= 12:
        return PageLayoutDecision(
            page_type="comic_panel_page",
            confidence=0.72,
            text_ratio=text_ratio,
            non_text_ratio=non_text_ratio,
            edge_density=edge_density,
            rect_count=rect_count,
            grid_score=grid_score,
        )

    if rect_count >= 2 and text_ratio >= 0.12:
        return PageLayoutDecision(
            page_type="activity_page",
            confidence=0.66,
            text_ratio=text_ratio,
            non_text_ratio=non_text_ratio,
            edge_density=edge_density,
            rect_count=rect_count,
            grid_score=grid_score,
        )

    if rect_count >= 1 and text_ratio <= 0.2:
        return PageLayoutDecision(
            page_type="framed_photo_page",
            confidence=0.62,
            text_ratio=text_ratio,
            non_text_ratio=non_text_ratio,
            edge_density=edge_density,
            rect_count=rect_count,
            grid_score=grid_score,
        )

    return PageLayoutDecision(
        page_type="mixed_page",
        confidence=0.5,
        text_ratio=text_ratio,
        non_text_ratio=non_text_ratio,
        edge_density=edge_density,
        rect_count=rect_count,
        grid_score=grid_score,
    )
