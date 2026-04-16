from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


PageArchetype = Literal[
    "comic_panel_page",
    "collage_card_page",
    "full_photo_page",
    "text_heavy_editorial_page",
    "mixed_page",
]


@dataclass(frozen=True)
class PageTypeInfo:
    archetype: PageArchetype
    confidence: float
    text_ratio: float
    non_text_ratio: float
    edge_density: float
    component_count: int


def detect_page_archetype(image_bgr: np.ndarray, non_text_mask: np.ndarray, text_mask: np.ndarray) -> PageTypeInfo:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges > 0)) / float(max(1, edges.size))

    text_ratio = float(np.count_nonzero(text_mask > 0)) / float(max(1, text_mask.size))
    non_text_ratio = float(np.count_nonzero(non_text_mask > 0)) / float(max(1, non_text_mask.size))

    contours, _ = cv2.findContours(non_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    component_count = len(contours)

    if text_ratio <= 0.14 and edge_density >= 0.02 and non_text_ratio >= 0.5:
        return PageTypeInfo("full_photo_page", 0.78, text_ratio, non_text_ratio, edge_density, component_count)
    if text_ratio >= 0.4 and non_text_ratio <= 0.45:
        return PageTypeInfo("text_heavy_editorial_page", 0.74, text_ratio, non_text_ratio, edge_density, component_count)
    if component_count >= 7 and 0.12 <= text_ratio <= 0.38:
        return PageTypeInfo("collage_card_page", 0.72, text_ratio, non_text_ratio, edge_density, component_count)
    if 3 <= component_count <= 8 and 0.12 <= text_ratio <= 0.35:
        return PageTypeInfo("comic_panel_page", 0.66, text_ratio, non_text_ratio, edge_density, component_count)
    return PageTypeInfo("mixed_page", 0.5, text_ratio, non_text_ratio, edge_density, component_count)

