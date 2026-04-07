import numpy as np

from src.backends.ocr_base import OCRLine
from src.config import ClassificationConfig, GroupingConfig
from src.pipeline.text_classification import classify_text_blocks
from src.pipeline.text_grouping import group_text_lines


def _line(text: str, bbox: tuple[int, int, int, int]) -> OCRLine:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    return OCRLine(
        text=text,
        confidence=0.9,
        bbox_px=bbox,
        line_height=float(h),
        line_width=float(w),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    )


def test_grouping_merges_lines_into_single_content_block() -> None:
    gray = np.zeros((600, 600), dtype=np.uint8)
    lines = [
        _line("line1", (50, 100, 420, 130)),
        _line("line2", (52, 140, 418, 170)),
        _line("line3", (49, 180, 422, 210)),
    ]
    blocks = group_text_lines(lines, gray, GroupingConfig())
    assert len(blocks) == 1
    assert len(blocks[0].regions_px) == 3


def test_headline_content_heuristic_smoke() -> None:
    gray = np.zeros((800, 800), dtype=np.uint8)
    lines = [
        _line("BASLIK", (100, 80, 500, 150)),
        _line("c1", (102, 220, 520, 250)),
        _line("c2", (104, 260, 522, 290)),
        _line("c3", (103, 300, 519, 330)),
    ]
    blocks = group_text_lines(lines, gray, GroupingConfig(max_vertical_gap_ratio=2.0, max_block_vertical_gap_ratio=2.0))
    classify_text_blocks(blocks, ClassificationConfig())
    roles = {b.role for b in blocks}
    assert "headline" in roles or "content" in roles


def test_prefix_headline_is_split_from_content_block() -> None:
    gray = np.zeros((1000, 1000), dtype=np.uint8)
    lines = [
        _line("Ektiğini biçer insan", (100, 100, 560, 155)),
        _line("Vaktiyle kimseciği olmayan ihtiyar bir kadıncağız...", (102, 210, 860, 248)),
        _line("Teyze, kanaatkâr ve mütevekkildi.", (102, 255, 830, 292)),
        _line("Avuç açmazdı kimseye, şükrederdi hep hâline.", (104, 298, 850, 335)),
    ]
    blocks = group_text_lines(lines, gray, GroupingConfig(max_vertical_gap_ratio=2.0, max_block_vertical_gap_ratio=2.0))
    classify_text_blocks(blocks, ClassificationConfig())
    roles = [b.role for b in blocks]
    assert "headline" in roles
    assert "content" in roles
