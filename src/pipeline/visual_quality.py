from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pipeline.layout_policy import LayoutBlockPlan
from src.utils.geometry import BBox, iou


@dataclass(frozen=True)
class _PlanMetrics:
    text_overlap: float
    edge_density: float
    entropy: float
    wr: float
    hr: float
    is_strip: bool
    score: float


def _entropy(gray_patch: np.ndarray) -> float:
    if gray_patch.size == 0:
        return 0.0
    hist = cv2.calcHist([gray_patch], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(1.0, float(hist.sum()))
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def _compute_metrics(image_bgr: np.ndarray, text_mask: np.ndarray, bbox: BBox) -> _PlanMetrics:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gpatch = gray[y1:y2, x1:x2]
    tpatch = text_mask[y1:y2, x1:x2]

    if gpatch.size == 0:
        return _PlanMetrics(1.0, 0.0, 0.0, 1.0, 1.0, True, -9.0)

    text_overlap = float(np.count_nonzero(tpatch > 0)) / float(max(1, tpatch.size))
    edges = cv2.Canny(gpatch, 80, 160)
    edge_density = float(np.count_nonzero(edges > 0)) / float(max(1, edges.size))
    ent = _entropy(gpatch)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    is_strip = (wr >= 0.84 and hr <= 0.30) or (hr >= 0.84 and wr <= 0.30)

    # Composite quality score for visual-rich crops.
    ent_norm = min(1.0, max(0.0, (ent - 2.0) / 4.0))
    score = (1.6 * edge_density) + (0.9 * ent_norm) + (0.9 * (1.0 - text_overlap))
    if is_strip:
        score -= 0.45
    return _PlanMetrics(
        text_overlap=text_overlap,
        edge_density=edge_density,
        entropy=ent,
        wr=wr,
        hr=hr,
        is_strip=is_strip,
        score=score,
    )


def refine_visual_plans(
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    plans: list[LayoutBlockPlan],
    page_type: str,
    min_keep: int = 1,
) -> list[LayoutBlockPlan]:
    if not plans:
        return []

    scored: list[tuple[LayoutBlockPlan, _PlanMetrics]] = []
    for p in plans:
        m = _compute_metrics(image_bgr=image_bgr, text_mask=text_mask, bbox=p.bbox)
        scored.append((p, m))

    # Hard reject rules for common failures (text-heavy strips / low-visual regions).
    accepted: list[tuple[LayoutBlockPlan, _PlanMetrics]] = []
    rejected: list[tuple[LayoutBlockPlan, _PlanMetrics]] = []
    text_overlap_limit = 0.58
    score_floor = 0.44
    if page_type in {"puzzle_page", "comic_panel_page", "activity_page"}:
        # Activity/comic pages include structured text inside visuals; slightly relax text cutoffs.
        text_overlap_limit = 0.66
        score_floor = 0.40

    for p, m in scored:
        hard_reject = False
        if p.visual_class != "background_photo_page":
            if m.is_strip and m.text_overlap >= 0.26:
                hard_reject = True
            if m.text_overlap >= text_overlap_limit and m.edge_density < 0.018 and m.entropy < 3.1:
                hard_reject = True
            if m.score < score_floor and m.text_overlap >= 0.34:
                hard_reject = True
        if hard_reject:
            rejected.append((p, m))
        else:
            accepted.append((p, m))

    # De-duplicate overlaps by score.
    accepted.sort(key=lambda x: x[1].score, reverse=True)
    dedup: list[tuple[LayoutBlockPlan, _PlanMetrics]] = []
    for p, m in accepted:
        if any(iou(p.bbox, q.bbox) > 0.72 for q, _ in dedup):
            continue
        dedup.append((p, m))

    # Guarantee minimum keep by rescuing strongest rejected candidates.
    if len(dedup) < min_keep:
        pool = sorted(rejected, key=lambda x: x[1].score, reverse=True)
        for p, m in pool:
            if any(iou(p.bbox, q.bbox) > 0.72 for q, _ in dedup):
                continue
            dedup.append((p, m))
            if len(dedup) >= min_keep:
                break

    # Keep ordering stable for downstream naming (top-to-bottom left-to-right).
    out: list[LayoutBlockPlan] = []
    dedup = sorted(dedup, key=lambda x: (x[0].bbox[1], x[0].bbox[0]))
    for idx, (p, m) in enumerate(dedup, start=1):
        reasons = list(p.review_reasons)
        needs_review = bool(p.needs_review)
        if m.score < 0.52 or m.text_overlap > 0.42:
            needs_review = True
            if "quality_low_confidence" not in reasons:
                reasons.append("quality_low_confidence")
        out.append(
            LayoutBlockPlan(
                bbox=p.bbox,
                visual_class=p.visual_class,
                confidence=p.confidence,
                needs_review=needs_review,
                review_reasons=reasons,
                page_side=p.page_side,
                panel_index=idx,
            )
        )
    return out
