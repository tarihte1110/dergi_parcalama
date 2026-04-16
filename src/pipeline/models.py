from __future__ import annotations

from dataclasses import dataclass, field

from src.backends.ocr_base import OCRLine
from src.utils.geometry import BBox


@dataclass
class TextBlock:
    block_id: str
    article_group_id: str
    role: str
    bbox_px: BBox
    regions_px: list[BBox]
    text: str
    lines: list[OCRLine] = field(default_factory=list)


@dataclass
class VisualBlock:
    block_id: str
    bbox_px: BBox
    area_ratio: float
    short_side_ratio: float
    crop_path: str
    visual_class: str = "framed_rectangular"
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    headline: str = ""
    headline_title: str = ""
    headline_byline: str = ""
    confidence: float = 0.0
    page_side: str = "single"
    panel_index: int = 1


@dataclass
class PageResult:
    image_path: str
    page_width: int
    page_height: int
    text_blocks: list[TextBlock]
    visual_blocks: list[VisualBlock]
