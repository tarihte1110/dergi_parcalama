from __future__ import annotations

from typing import Iterable


BBox = tuple[int, int, int, int]


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(bbox: BBox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def bbox_union(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("bbox_union requires at least one box")
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


def intersection_area(a: BBox, b: BBox) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    return inter / float(bbox_area(a) + bbox_area(b) - inter)


def horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    overlap = max(0, right - left)
    min_w = max(1, min(a[2] - a[0], b[2] - b[0]))
    return overlap / min_w


def vertical_gap(a: BBox, b: BBox) -> int:
    if b[1] >= a[3]:
        return b[1] - a[3]
    if a[1] >= b[3]:
        return a[1] - b[3]
    return 0


def normalize_bbox_1000(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        int(round((x1 / max(1, width)) * 1000)),
        int(round((y1 / max(1, height)) * 1000)),
        int(round((x2 / max(1, width)) * 1000)),
        int(round((y2 / max(1, height)) * 1000)),
    )


def denormalize_bbox_1000(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        int(round((x1 / 1000.0) * width)),
        int(round((y1 / 1000.0) * height)),
        int(round((x2 / 1000.0) * width)),
        int(round((y2 / 1000.0) * height)),
    )


def sort_bboxes_yx(boxes: Iterable[BBox]) -> list[BBox]:
    return sorted(boxes, key=lambda b: (b[1], b[0], b[3], b[2]))
