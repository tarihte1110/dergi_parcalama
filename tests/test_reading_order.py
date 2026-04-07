from src.backends.ocr_base import OCRLine
from src.config import GroupingConfig
from src.pipeline.models import TextBlock
from src.pipeline.reading_order import order_blocks_for_reading, order_lines_in_block


def _line(text: str, bbox: tuple[int, int, int, int]) -> OCRLine:
    x1, y1, x2, y2 = bbox
    return OCRLine(
        text=text,
        confidence=0.9,
        bbox_px=bbox,
        line_height=float(y2 - y1),
        line_width=float(x2 - x1),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    )


def test_order_lines_in_block_reads_left_column_then_right() -> None:
    cfg = GroupingConfig()
    lines = [
        _line("R1", (420, 100, 520, 130)),
        _line("L1", (60, 80, 200, 110)),
        _line("R2", (420, 140, 520, 170)),
        _line("L2", (60, 120, 200, 150)),
    ]
    ordered = order_lines_in_block(lines, cfg)
    assert [x.text for x in ordered] == ["L1", "L2", "R1", "R2"]


def test_order_blocks_for_reading_row_major() -> None:
    cfg = GroupingConfig()
    blocks = [
        TextBlock("t1", "a1", "content", (500, 80, 700, 200), [(500, 80, 700, 200)], "B", []),
        TextBlock("t2", "a2", "content", (60, 90, 260, 230), [(60, 90, 260, 230)], "A", []),
        TextBlock("t3", "a3", "content", (80, 420, 300, 560), [(80, 420, 300, 560)], "C", []),
    ]
    ordered = order_blocks_for_reading(blocks, cfg)
    assert [b.text for b in ordered] == ["A", "B", "C"]
