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

    heights = [max(1, b.bbox_px[3] - b.bbox_px[1]) for b in blocks]
    median_h = float(np.median(heights))
    band = max(24.0, cfg.block_row_band_ratio * median_h)
    return sorted(
        blocks,
        key=lambda b: (
            int(b.bbox_px[1] // band),
            b.bbox_px[0],
            b.bbox_px[1],
            b.bbox_px[2],
        ),
    )
