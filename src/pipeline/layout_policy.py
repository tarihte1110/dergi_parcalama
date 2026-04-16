from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.page_layout_classifier import PageLayoutDecision
from src.utils.geometry import BBox, bbox_area, iou


@dataclass(frozen=True)
class LayoutBlockPlan:
    bbox: BBox
    visual_class: str
    confidence: float
    needs_review: bool
    review_reasons: list[str]
    page_side: str
    panel_index: int


def _is_band_like(bbox: BBox, page_size: tuple[int, int]) -> bool:
    h, w = page_size
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    ar = (bw * bh) / float(max(1, w * h))
    return ((wr >= 0.92 and hr <= 0.42) or (hr >= 0.92 and wr <= 0.42)) and ar >= 0.15


def _side_of_bbox(bbox: BBox, page_size: tuple[int, int]) -> str:
    _, w = page_size
    cx = (bbox[0] + bbox[2]) / 2.0
    return "left" if cx <= (w / 2.0) else "right"


def _filter_and_rank_boxes(
    boxes: list[BBox],
    page_size: tuple[int, int],
    max_keep: int = 2,
) -> list[BBox]:
    h, w = page_size
    page_area = float(max(1, h * w))
    scored: list[tuple[float, BBox]] = []
    for b in boxes:
        ar = bbox_area(b) / page_area
        if ar < 0.05:
            continue
        if _is_band_like(b, page_size):
            continue
        bw = max(1, b[2] - b[0])
        bh = max(1, b[3] - b[1])
        wr = bw / float(max(1, w))
        hr = bh / float(max(1, h))
        shape_bonus = 0.0
        if 0.18 <= wr <= 0.82 and 0.18 <= hr <= 0.95:
            shape_bonus += 0.2
        score = ar + shape_bonus
        scored.append((score, b))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[BBox] = []
    for _, b in scored:
        if any(iou(b, k) > 0.7 for k in out):
            continue
        out.append(b)
        if len(out) >= max_keep:
            break
    return sorted(out, key=lambda bb: (bb[1], bb[0]))


def _build_plan_for_boxes(
    boxes: list[BBox],
    page_size: tuple[int, int],
    page_type: str,
    base_conf: float,
) -> list[LayoutBlockPlan]:
    h, w = page_size
    if not boxes:
        return []
    out: list[LayoutBlockPlan] = []
    for idx, b in enumerate(boxes, start=1):
        ar = bbox_area(b) / float(max(1, h * w))
        if page_type in {"comic_panel_page", "activity_page"}:
            klass = "comic_panel_page"
        elif page_type == "puzzle_page":
            klass = "puzzle_page"
        elif page_type == "framed_photo_page":
            klass = "framed_photo"
        elif page_type == "background_photo_page":
            klass = "background_photo_page"
        else:
            if ar >= 0.45:
                klass = "background_photo_page"
            elif ar >= 0.12:
                klass = "framed_photo"
            else:
                klass = "corner_illustration"
        side = _side_of_bbox(b, (h, w)) if (w / float(max(1, h))) >= 1.35 else "single"
        out.append(
            LayoutBlockPlan(
                bbox=b,
                visual_class=klass,
                confidence=round(base_conf, 3),
                needs_review=False,
                review_reasons=[],
                page_side=side,
                panel_index=idx,
            )
        )
    return out


def _select_puzzle_boxes(
    boxes: list[BBox],
    page_size: tuple[int, int],
    max_keep: int = 4,
) -> list[BBox]:
    h, w = page_size
    if not boxes:
        return []
    page_area = float(max(1, h * w))
    filtered: list[BBox] = []
    for b in boxes:
        ar = bbox_area(b) / page_area
        if ar < 0.045:
            continue
        if _is_band_like(b, page_size):
            continue
        filtered.append(b)
    if not filtered:
        return []

    by_area = sorted(filtered, key=lambda b: bbox_area(b), reverse=True)
    out: list[BBox] = []

    # Ensure both halves are represented on spread pages.
    if (w / float(max(1, h))) >= 1.35:
        left = [b for b in by_area if _side_of_bbox(b, page_size) == "left"]
        right = [b for b in by_area if _side_of_bbox(b, page_size) == "right"]
        if left:
            out.append(left[0])
        if right and (not out or iou(right[0], out[0]) < 0.7):
            out.append(right[0])

    # Add quadrant diversity.
    quadrant_best: dict[tuple[int, int], BBox] = {}
    for b in by_area:
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        qx = 0 if cx <= (w / 2.0) else 1
        qy = 0 if cy <= (h / 2.0) else 1
        key = (qx, qy)
        prev = quadrant_best.get(key)
        if prev is None or bbox_area(b) > bbox_area(prev):
            quadrant_best[key] = b

    for b in sorted(quadrant_best.values(), key=lambda bb: bbox_area(bb), reverse=True):
        if len(out) >= max_keep:
            break
        if any(iou(b, k) > 0.68 for k in out):
            continue
        out.append(b)

    # Fill remaining by area ranking.
    for b in by_area:
        if len(out) >= max_keep:
            break
        if any(iou(b, k) > 0.68 for k in out):
            continue
        out.append(b)
    return sorted(out, key=lambda bb: (bb[1], bb[0]))


def apply_layout_policy(
    page_decision: PageLayoutDecision,
    page_size: tuple[int, int],
    pattern_boxes: list[BBox],
    heuristic_boxes: list[BBox],
    force_fullpage: bool = False,
) -> list[LayoutBlockPlan]:
    h, w = page_size
    if force_fullpage:
        return [
            LayoutBlockPlan(
                bbox=(0, 0, w, h),
                visual_class="background_photo_page",
                confidence=1.0,
                needs_review=False,
                review_reasons=[],
                page_side="single",
                panel_index=1,
            )
        ]

    source = pattern_boxes if pattern_boxes else heuristic_boxes
    if not source:
        return []

    # Type-specific keep limits.
    keep_limit = 2
    if page_decision.page_type == "background_photo_page":
        keep_limit = 1
    if page_decision.page_type in {"comic_panel_page", "activity_page"}:
        keep_limit = 2
    if page_decision.page_type == "puzzle_page":
        keep_limit = 4

    if page_decision.page_type == "puzzle_page":
        selected = _select_puzzle_boxes(source, page_size=page_size, max_keep=keep_limit)
    else:
        selected = _filter_and_rank_boxes(source, page_size=page_size, max_keep=keep_limit)
    if not selected:
        return []
    return _build_plan_for_boxes(
        selected,
        page_size=page_size,
        page_type=page_decision.page_type,
        base_conf=page_decision.confidence,
    )
