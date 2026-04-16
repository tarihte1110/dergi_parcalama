from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backends.ocr_base import OCRLine
from src.config import GroupingConfig
from src.pipeline.models import TextBlock
from src.utils.geometry import horizontal_overlap_ratio


@dataclass
class _Column:
    min_x: int
    max_x: int
    lines: list[OCRLine]

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0


def order_lines_in_block(lines: list[OCRLine], cfg: GroupingConfig) -> list[OCRLine]:
    if len(lines) <= 1:
        return lines

    median_h = float(np.median([l.line_height for l in lines]))
    align_tol = max(8.0, cfg.line_column_align_ratio * median_h)
    sorted_input = sorted(lines, key=lambda l: (l.bbox_px[0], l.bbox_px[1]))
    columns: list[_Column] = []

    for line in sorted_input:
        lx1, _, lx2, _ = line.bbox_px
        best_idx = -1
        best_score = -1.0
        for i, col in enumerate(columns):
            col_box = (col.min_x, line.bbox_px[1], col.max_x, line.bbox_px[3])
            overlap = horizontal_overlap_ratio(line.bbox_px, col_box)
            center_dx = abs(((lx1 + lx2) / 2.0) - col.center_x)
            if overlap > 0.12 or center_dx <= align_tol:
                score = overlap - (center_dx / max(1.0, align_tol))
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_idx == -1:
            columns.append(_Column(min_x=lx1, max_x=lx2, lines=[line]))
        else:
            col = columns[best_idx]
            col.lines.append(line)
            col.min_x = min(col.min_x, lx1)
            col.max_x = max(col.max_x, lx2)

    columns.sort(key=lambda c: c.min_x)
    ordered: list[OCRLine] = []
    for col in columns:
        ordered.extend(sorted(col.lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0])))
    return ordered


def order_blocks_for_reading(blocks: list[TextBlock], cfg: GroupingConfig) -> list[TextBlock]:
    if len(blocks) <= 1:
        return blocks

    def _order_single_page(items: list[TextBlock]) -> list[TextBlock]:
        # Keep block order primarily top-down so a started block is completed
        # before jumping to a later-starting block.
        heights = [max(1, b.bbox_px[3] - b.bbox_px[1]) for b in items]
        median_h = float(np.median(heights))
        band = min(120.0, max(24.0, cfg.block_row_band_ratio * median_h))
        return sorted(
            items,
            key=lambda b: (
                int(b.bbox_px[1] // band),
                b.bbox_px[0],
                b.bbox_px[1],
                b.bbox_px[2],
            ),
        )

    min_x = min(b.bbox_px[0] for b in blocks)
    max_x = max(b.bbox_px[2] for b in blocks)
    min_y = min(b.bbox_px[1] for b in blocks)
    max_y = max(b.bbox_px[3] for b in blocks)
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    aspect = width / float(height)

    # Spread-aware ordering:
    # if page looks like two-page spread, finish the left page first,
    # then continue with the right page; each side keeps top-down, left-right order.
    if aspect >= 1.35:
        mid_x = (min_x + max_x) / 2.0
        left: list[TextBlock] = []
        right: list[TextBlock] = []
        for b in blocks:
            cx = (b.bbox_px[0] + b.bbox_px[2]) / 2.0
            if cx <= mid_x:
                left.append(b)
            else:
                right.append(b)
        if left and right:
            return _order_single_page(left) + _order_single_page(right)

    return _order_single_page(blocks)
