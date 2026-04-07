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


@dataclass
class PageResult:
    image_path: str
    page_width: int
    page_height: int
    text_blocks: list[TextBlock]
    visual_blocks: list[VisualBlock]
