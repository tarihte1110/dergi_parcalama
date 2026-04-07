from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.backends.ocr_base import OCRLine
from src.config import GroupingConfig
from src.pipeline.models import TextBlock
from src.pipeline.reading_order import order_lines_in_block
from src.pipeline.text_postprocess import clean_ocr_text
from src.utils.geometry import BBox, bbox_union, horizontal_overlap_ratio, vertical_gap


@dataclass
class _DisjointSet:
    parent: list[int]

    @classmethod
    def with_size(cls, n: int) -> "_DisjointSet":
        return cls(parent=list(range(n)))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _edge_density(gray: np.ndarray, bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    edges = cv2.Canny(patch, 100, 200)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _is_barrier_between(a: BBox, b: BBox, gray: np.ndarray, cfg: GroupingConfig, median_h: float) -> bool:
    gap = vertical_gap(a, b)
    if gap <= cfg.barrier_gap_ratio * median_h:
        return False

    x1 = int(min(a[0], b[0]))
    x2 = int(max(a[2], b[2]))
    y1 = int(min(a[3], b[3]))
    y2 = int(max(a[1], b[1]))
    if y2 <= y1 or x2 <= x1:
        return False

    density = _edge_density(gray, (x1, y1, x2, y2))
    return density >= cfg.barrier_edge_density


def _should_connect(a: OCRLine, b: OCRLine, gray: np.ndarray, cfg: GroupingConfig, median_h: float) -> bool:
    box_a = a.bbox_px
    box_b = b.bbox_px
    if box_a[1] > box_b[1]:
        box_a, box_b = box_b, box_a

    height_ratio = max(a.line_height, b.line_height) / max(1.0, min(a.line_height, b.line_height))
    if height_ratio > cfg.max_line_height_ratio:
        return False

    gap = vertical_gap(box_a, box_b)
    if gap / max(1.0, median_h) > cfg.max_vertical_gap_ratio:
        return False

    h_overlap = horizontal_overlap_ratio(box_a, box_b)
    left_delta = abs(box_a[0] - box_b[0]) / max(1.0, median_h)
    center_delta = abs(a.center[0] - b.center[0]) / max(1.0, median_h)

    alignment_ok = (
        h_overlap >= cfg.min_horizontal_overlap_ratio
        or left_delta <= cfg.max_alignment_delta_ratio
        or center_delta <= cfg.max_alignment_delta_ratio
    )
    if not alignment_ok:
        return False

    if center_delta > cfg.max_reading_flow_dx_ratio:
        return False

    if _is_barrier_between(box_a, box_b, gray, cfg, median_h):
        return False

    return True


def _merge_lines_to_block(lines: list[OCRLine], idx: int) -> TextBlock:
    sorted_lines = sorted(lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))
    regions = [l.bbox_px for l in sorted_lines]
    text = clean_ocr_text(" ".join(line.text for line in sorted_lines if line.text.strip()))
    block_bbox = bbox_union(regions)
    return TextBlock(
        block_id=f"t{idx}",
        article_group_id="",
        role="other_text",
        bbox_px=block_bbox,
        regions_px=regions,
        text=text,
        lines=sorted_lines,
    )


def _merge_blocks_once(blocks: list[TextBlock], gray: np.ndarray, cfg: GroupingConfig, median_h: float) -> list[TextBlock]:
    if len(blocks) < 2:
        return blocks

    dsu = _DisjointSet.with_size(len(blocks))
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a = blocks[i].bbox_px
            b = blocks[j].bbox_px
            if a[1] > b[1]:
                a, b = b, a

            gap = vertical_gap(a, b) / max(1.0, median_h)
            if gap > cfg.max_block_vertical_gap_ratio:
                continue

            h_overlap = horizontal_overlap_ratio(blocks[i].bbox_px, blocks[j].bbox_px)
            left_delta = abs(blocks[i].bbox_px[0] - blocks[j].bbox_px[0]) / max(1.0, median_h)
            if h_overlap < cfg.min_horizontal_overlap_ratio and left_delta > cfg.max_alignment_delta_ratio:
                continue

            if _is_barrier_between(blocks[i].bbox_px, blocks[j].bbox_px, gray, cfg, median_h):
                continue

            dsu.union(i, j)

    groups: dict[int, list[TextBlock]] = {}
    for i, block in enumerate(blocks):
        root = dsu.find(i)
        groups.setdefault(root, []).append(block)

    merged: list[TextBlock] = []
    for items in groups.values():
        all_lines: list[OCRLine] = []
        for b in items:
            all_lines.extend(b.lines)
        merged.append(_merge_lines_to_block(all_lines, idx=0))
    return merged


def group_text_lines(lines: list[OCRLine], gray: np.ndarray, cfg: GroupingConfig) -> list[TextBlock]:
    if not lines:
        return []

    median_h = float(np.median([l.line_height for l in lines]))
    dsu = _DisjointSet.with_size(len(lines))

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _should_connect(lines[i], lines[j], gray, cfg, median_h):
                dsu.union(i, j)

    groups: dict[int, list[OCRLine]] = {}
    for idx, line in enumerate(lines):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(line)

    blocks = []
    for i, group_lines in enumerate(groups.values(), start=1):
        ordered_group_lines = order_lines_in_block(group_lines, cfg)
        blocks.append(_merge_lines_to_block(ordered_group_lines, i))
    blocks = _merge_blocks_once(blocks, gray, cfg, median_h)
    blocks = _absorb_short_blocks(blocks, cfg, median_h)

    blocks = sorted(blocks, key=lambda b: (b.bbox_px[1], b.bbox_px[0]))
    for i, block in enumerate(blocks, start=1):
        block.block_id = f"t{i}"
        block.bbox_px = bbox_union(block.regions_px)
    return blocks


def _absorb_short_blocks(blocks: list[TextBlock], cfg: GroupingConfig, median_h: float) -> list[TextBlock]:
    if len(blocks) < 2:
        return blocks
    survivors: list[TextBlock] = []
    absorbed = [False] * len(blocks)

    def block_center(b: TextBlock) -> tuple[float, float]:
        x1, y1, x2, y2 = b.bbox_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    for i, block in enumerate(blocks):
        if absorbed[i]:
            continue
        text_len = len(block.text.strip())
        if text_len > cfg.absorb_short_text_max_chars:
            survivors.append(block)
            continue

        cx, cy = block_center(block)
        nearest_idx = -1
        nearest_dist = 1e18
        for j, other in enumerate(blocks):
            if i == j or absorbed[j]:
                continue
            ocx, ocy = block_center(other)
            dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = j

        if nearest_idx == -1 or nearest_dist / max(1.0, median_h) > cfg.absorb_max_center_distance_ratio:
            survivors.append(block)
            continue

        target = blocks[nearest_idx]
        target.lines.extend(block.lines)
        merged = _merge_lines_to_block(target.lines, idx=0)
        blocks[nearest_idx] = merged
        absorbed[i] = True

    final_blocks = [b for k, b in enumerate(blocks) if not absorbed[k]]
    return final_blocks
