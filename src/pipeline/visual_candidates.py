from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from src.config import VisualFilterConfig
from src.utils.geometry import BBox, bbox_area, bbox_union, clip_bbox, iou
from src.utils.image_ops import ratio_to_kernel


@dataclass(frozen=True)
class CandidateMetrics:
    bbox: BBox
    area_ratio: float
    short_side_ratio: float
    aspect_ratio: float
    edge_density: float
    fill_ratio: float
    text_overlap_ratio: float
    entropy: float


@dataclass(frozen=True)
class RejectedCandidate:
    bbox: BBox
    reason: str
    area_ratio: float


VisualClass = Literal[
    "framed_rectangular",
    "freeform_illustrations",
    "collage_cards",
    "comic_panels",
    "full_compositions",
    "ambiguous_review",
]


@dataclass(frozen=True)
class VisualDecision:
    metrics: CandidateMetrics
    visual_class: VisualClass
    needs_review: bool
    review_reasons: list[str]


def _entropy(gray_patch: np.ndarray) -> float:
    if gray_patch.size == 0:
        return 0.0
    hist = cv2.calcHist([gray_patch], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(1.0, hist.sum())
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def _compute_metrics(
    image_bgr: np.ndarray,
    contour: np.ndarray,
    bbox: BBox,
    text_mask: np.ndarray | None = None,
) -> CandidateMetrics:
    h, w = image_bgr.shape[:2]
    area = bbox_area(bbox)
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    aspect = max(bw / bh, bh / bw)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = bbox
    patch = gray[y1:y2, x1:x2]
    edges = cv2.Canny(patch, 80, 160) if patch.size else np.zeros((1, 1), dtype=np.uint8)
    edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))

    contour_area = float(cv2.contourArea(contour))
    fill_ratio = contour_area / float(max(1, area))
    entropy = _entropy(patch)
    if text_mask is None:
        text_overlap = 0.0
    else:
        tpatch = text_mask[y1:y2, x1:x2]
        text_overlap = float(np.count_nonzero(tpatch > 0)) / float(max(1, tpatch.size))

    return CandidateMetrics(
        bbox=bbox,
        area_ratio=area / float(h * w),
        short_side_ratio=min(bw / float(w), bh / float(h)),
        aspect_ratio=aspect,
        edge_density=edge_density,
        fill_ratio=fill_ratio,
        text_overlap_ratio=text_overlap,
        entropy=entropy,
    )


def _touches_edges(bbox: BBox, w: int, h: int, tol: int = 2) -> int:
    x1, y1, x2, y2 = bbox
    cnt = 0
    if x1 <= tol:
        cnt += 1
    if y1 <= tol:
        cnt += 1
    if x2 >= w - tol:
        cnt += 1
    if y2 >= h - tol:
        cnt += 1
    return cnt


def keep_candidate(metrics: CandidateMetrics, cfg: VisualFilterConfig, page_size: tuple[int, int]) -> tuple[bool, str]:
    x1, y1, x2, y2 = metrics.bbox
    bw = x2 - x1
    bh = y2 - y1
    h, w = page_size
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    if bw < cfg.min_visual_width_px or bh < cfg.min_visual_height_px:
        return False, "too_small_abs"
    if metrics.area_ratio < cfg.min_visual_area_ratio:
        return False, "too_small_ratio"
    if metrics.area_ratio > cfg.max_visual_area_ratio:
        return False, "too_large_ratio"
    if metrics.short_side_ratio < cfg.min_visual_short_side_ratio:
        return False, "too_thin_relative"
    if metrics.aspect_ratio > cfg.max_aspect_ratio:
        return False, "too_long_strip"
    if wr >= 0.97 and hr >= 0.97:
        return False, "full_page_like"
    is_canvas_strip = (wr > cfg.strip_span_ratio and hr < cfg.strip_other_ratio) or (
        hr > cfg.strip_span_ratio and wr < cfg.strip_other_ratio
    )
    if is_canvas_strip:
        # Rescue real visual bands (e.g., wide story illustrations) that still carry strong visual signal.
        strip_rescue = (
            metrics.area_ratio >= 0.08
            and metrics.area_ratio <= 0.45
            and metrics.edge_density >= max(0.008, cfg.min_edge_density * 1.2)
            and metrics.entropy >= 3.4
            and metrics.fill_ratio >= 0.18
        )
        if not strip_rescue:
            return False, "too_canvas_strip"
    edge_touches = _touches_edges(metrics.bbox, w, h)
    if edge_touches >= 3 and metrics.area_ratio > cfg.edge_touch_reject_area_ratio:
        edge_touch_rescue = (
            metrics.edge_density >= cfg.edge_touch_rescue_min_edge_density
            and metrics.entropy >= cfg.edge_touch_rescue_min_entropy
            and metrics.area_ratio <= cfg.edge_touch_rescue_max_area_ratio
        )
        if not edge_touch_rescue:
            return False, "edge_touch_large"
    if metrics.text_overlap_ratio > cfg.max_text_overlap_ratio:
        text_on_image_rescue = (
            metrics.area_ratio >= cfg.text_heavy_rescue_min_area_ratio
            and metrics.edge_density >= cfg.text_heavy_rescue_min_edge_density
            and metrics.entropy >= cfg.text_heavy_rescue_min_entropy
        )
        if not text_on_image_rescue:
            return False, "text_heavy_region"
    if metrics.edge_density < cfg.min_edge_density:
        return False, "low_edge_density"
    if metrics.area_ratio < cfg.small_region_high_quality_area_ratio:
        if metrics.edge_density < cfg.small_region_min_edge_density:
            return False, "small_low_edge"
        if metrics.entropy < cfg.small_region_min_entropy:
            return False, "small_low_entropy"
    if metrics.fill_ratio < cfg.min_fill_ratio:
        return False, "low_fill_ratio"
    if metrics.entropy < cfg.min_entropy:
        return False, "low_entropy"
    return True, "ok"


def _merge_overlaps(boxes: list[BBox], iou_threshold: float) -> list[BBox]:
    if not boxes:
        return []
    used = [False] * len(boxes)
    merged: list[BBox] = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        current = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                if iou(current, boxes[j]) >= iou_threshold:
                    current = bbox_union([current, boxes[j]])
                    used[j] = True
                    changed = True
        merged.append(current)
    return merged


def _axis_gap(a1: int, a2: int, b1: int, b2: int) -> int:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def _cross_overlap_ratio(a1: int, a2: int, b1: int, b2: int) -> float:
    ov = max(0, min(a2, b2) - max(a1, b1))
    mn = max(1, min(a2 - a1, b2 - b1))
    return ov / float(mn)


def _intersection_area(a: BBox, b: BBox) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _should_merge_fragments(a: BBox, b: BBox, page_min_dim: int, cfg: VisualFilterConfig) -> bool:
    gap_limit = max(3, int(round(page_min_dim * cfg.fragment_merge_gap_ratio)))
    x_gap = _axis_gap(a[0], a[2], b[0], b[2])
    y_gap = _axis_gap(a[1], a[3], b[1], b[3])
    if x_gap > gap_limit and y_gap > gap_limit:
        return False

    y_cross = _cross_overlap_ratio(a[1], a[3], b[1], b[3])
    x_cross = _cross_overlap_ratio(a[0], a[2], b[0], b[2])
    if y_cross < cfg.fragment_merge_min_cross_overlap and x_cross < cfg.fragment_merge_min_cross_overlap:
        return False
    return True


def _merge_fragmented_boxes(boxes: list[BBox], page_size: tuple[int, int], cfg: VisualFilterConfig) -> list[BBox]:
    if len(boxes) < 2:
        return boxes
    h, w = page_size
    page_area = float(max(1, h * w))
    page_min_dim = min(h, w)

    current = list(boxes)
    changed = True
    while changed:
        changed = False
        used = [False] * len(current)
        merged: list[BBox] = []
        for i in range(len(current)):
            if used[i]:
                continue
            cur = current[i]
            used[i] = True
            expanded = True
            while expanded:
                expanded = False
                for j in range(len(current)):
                    if used[j]:
                        continue
                    other = current[j]
                    if not _should_merge_fragments(cur, other, page_min_dim, cfg):
                        continue
                    uni = bbox_union([cur, other])
                    if (bbox_area(uni) / page_area) > cfg.fragment_merge_max_union_area_ratio:
                        continue
                    cur = uni
                    used[j] = True
                    expanded = True
                    changed = True
            merged.append(cur)
        current = merged
    return current


def _bbox_shape_ratios(bbox: BBox, page_size: tuple[int, int]) -> tuple[float, float, float]:
    h, w = page_size
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    ar = bbox_area(bbox) / float(max(1, h * w))
    return wr, hr, ar


def _is_full_width_band(bbox: BBox, page_size: tuple[int, int]) -> bool:
    wr, hr, ar = _bbox_shape_ratios(bbox, page_size)
    horizontal_band = wr >= 0.9 and hr <= 0.45 and ar >= 0.18
    vertical_band = hr >= 0.9 and wr <= 0.45 and ar >= 0.18
    return horizontal_band or vertical_band


def _extract_center_focus_boxes(
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    page_size: tuple[int, int],
) -> list[BBox]:
    h, w = page_size
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160) > 0
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.5)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.5)
    std = np.sqrt(np.maximum(0.0, mean_sq - (mean * mean)))
    sat = hsv[:, :, 1]

    text_dilate = cv2.dilate(
        (text_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ) > 0

    feat = (edges | (std >= 10.0) | (sat >= 24)) & (~text_dilate)
    if not np.any(feat):
        return []

    # Keep search around the main composition region.
    x1 = int(round(w * 0.05))
    x2 = int(round(w * 0.95))
    y1 = int(round(h * 0.05))
    y2 = int(round(h * 0.95))
    roi = feat[y1:y2, x1:x2].astype(np.uint8) * 255
    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    page_area = float(max(1, h * w))
    center_x = w / 2.0
    center_y = h / 2.0
    ranked: list[tuple[float, BBox]] = []
    for contour in contours:
        rx, ry, bw, bh = cv2.boundingRect(contour)
        bbox = clip_bbox((x1 + rx, y1 + ry, x1 + rx + bw, y1 + ry + bh), w, h)
        wr, hr, ar = _bbox_shape_ratios(bbox, (h, w))
        if ar < 0.045 or ar > 0.62:
            continue
        if _is_full_width_band(bbox, (h, w)):
            continue
        bx1, by1, bx2, by2 = bbox
        patch = gray[by1:by2, bx1:bx2]
        if patch.size == 0:
            continue
        entropy = _entropy(patch)
        edge_patch = cv2.Canny(patch, 80, 160)
        edge_den = float(np.count_nonzero(edge_patch > 0)) / float(max(1, edge_patch.size))
        tpatch = text_mask[by1:by2, bx1:bx2]
        text_overlap = float(np.count_nonzero(tpatch > 0)) / float(max(1, tpatch.size))
        if text_overlap > 0.28:
            continue
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5 / max(1.0, (w**2 + h**2) ** 0.5)
        score = (
            1.4 * min(1.0, edge_den / 0.03)
            + 1.2 * min(1.0, entropy / 5.0)
            + 0.6 * min(1.0, ar / 0.35)
            - 0.9 * min(1.0, text_overlap / 0.28)
            - 0.8 * dist
        )
        ranked.append((score, bbox))

    if not ranked:
        return []
    ranked.sort(key=lambda x: x[0], reverse=True)
    out: list[BBox] = []
    for _, bbox in ranked:
        if any(iou(bbox, b) > 0.7 for b in out):
            continue
        out.append(bbox)
        if len(out) >= 2:
            break
    return out


def _coalesce_visual_panels(
    candidates: list[CandidateMetrics],
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
    page_size: tuple[int, int],
) -> list[CandidateMetrics]:
    if not candidates:
        return []
    h, w = page_size
    page_area = float(max(1, h * w))
    page_min_dim = min(h, w)
    boxes = [c.bbox for c in candidates]

    # Remove near-duplicate contained boxes by keeping the larger one.
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            inter = _intersection_area(boxes[i], boxes[j])
            min_area = float(max(1, min(bbox_area(boxes[i]), bbox_area(boxes[j]))))
            if (inter / min_area) < cfg.panel_drop_small_inside_ratio:
                continue
            if bbox_area(boxes[i]) >= bbox_area(boxes[j]):
                keep[j] = False
            else:
                keep[i] = False
                break
    current = [boxes[i] for i in range(len(boxes)) if keep[i]]

    gap_limit = max(4, int(round(page_min_dim * cfg.panel_merge_gap_ratio)))
    changed = True
    while changed and len(current) > 1:
        changed = False
        used = [False] * len(current)
        merged: list[BBox] = []
        for i in range(len(current)):
            if used[i]:
                continue
            cur = current[i]
            used[i] = True
            expanded = True
            while expanded:
                expanded = False
                for j in range(len(current)):
                    if used[j]:
                        continue
                    other = current[j]
                    x_gap = _axis_gap(cur[0], cur[2], other[0], other[2])
                    y_gap = _axis_gap(cur[1], cur[3], other[1], other[3])
                    x_cross = _cross_overlap_ratio(cur[0], cur[2], other[0], other[2])
                    y_cross = _cross_overlap_ratio(cur[1], cur[3], other[1], other[3])
                    should_merge = False
                    if x_gap <= gap_limit and y_cross >= cfg.panel_merge_min_cross_overlap:
                        should_merge = True
                    if y_gap <= gap_limit and x_cross >= cfg.panel_merge_min_cross_overlap:
                        should_merge = True
                    if iou(cur, other) >= 0.02:
                        should_merge = True
                    if not should_merge:
                        continue
                    uni = bbox_union([cur, other])
                    u_wr, u_hr, u_ar = _bbox_shape_ratios(uni, (h, w))
                    # Prevent coalescing into page-wide horizontal/vertical bands.
                    if (u_wr >= 0.95 and u_hr <= 0.82 and u_ar >= 0.24) or (u_hr >= 0.95 and u_wr <= 0.82 and u_ar >= 0.24):
                        continue
                    if (bbox_area(uni) / page_area) > cfg.panel_merge_max_union_area_ratio:
                        continue
                    cur = uni
                    used[j] = True
                    expanded = True
                    changed = True
            merged.append(cur)
        current = merged

    out: list[CandidateMetrics] = []
    for bbox in current:
        x1, y1, x2, y2 = bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        if m.area_ratio < cfg.panel_min_area_ratio and len(current) > 2:
            continue
        if m.text_overlap_ratio > cfg.panel_text_heavy_drop_ratio and m.edge_density < cfg.panel_text_heavy_max_edge_density:
            continue
        if (
            m.area_ratio <= cfg.small_textlike_max_area_ratio
            and m.entropy <= cfg.small_textlike_max_entropy
            and m.edge_density <= cfg.small_textlike_max_edge_density
        ):
            continue
        if m.area_ratio > 0.97:
            continue
        out.append(m)
    return out


def _split_huge_wide_box(
    bbox: BBox,
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    page_size: tuple[int, int],
) -> list[BBox]:
    h, w = page_size
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return [bbox]
    area_ratio = bbox_area(bbox) / float(max(1, h * w))
    wr = bw / float(max(1, w))
    if area_ratio < 0.68 or wr < 0.9:
        return [bbox]

    region = (non_text_mask[y1:y2, x1:x2] > 0) & (text_mask[y1:y2, x1:x2] == 0)
    if not np.any(region):
        return [bbox]
    profile = region.mean(axis=0)
    l = int(round(0.35 * bw))
    r = int(round(0.65 * bw))
    if r <= l + 5:
        return [bbox]
    mid_slice = profile[l:r]
    valley_rel = int(np.argmin(mid_slice))
    valley = l + valley_rel
    if profile[valley] > 0.22:
        return [bbox]

    left = region[:, :valley]
    right = region[:, valley:]
    if left.shape[1] < int(0.22 * bw) or right.shape[1] < int(0.22 * bw):
        return [bbox]
    out: list[BBox] = []
    for side, off in [(left, 0), (right, valley)]:
        ys, xs = np.where(side)
        if ys.size == 0:
            continue
        bx1 = x1 + off + int(xs.min())
        bx2 = x1 + off + int(xs.max()) + 1
        by1 = y1 + int(ys.min())
        by2 = y1 + int(ys.max()) + 1
        pad = max(6, int(round(min(h, w) * 0.01)))
        out.append(clip_bbox((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), w, h))
    return out if len(out) >= 2 else [bbox]


def _extract_wide_half_panels(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
    page_size: tuple[int, int],
) -> list[BBox]:
    h, w = page_size
    if (w / float(max(1, h))) < 1.35:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.0)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)
    sat = hsv[:, :, 1]
    edges = cv2.Canny(gray, 80, 160) > 0
    feature = (std >= cfg.half_rescue_texture_std_threshold) | (sat >= cfg.half_rescue_saturation_threshold) | edges
    # Half rescue must be text-agnostic for comic pages with speech bubbles.
    # Build a soft background mask and keep all visual-rich regions regardless of OCR mask holes.
    soft_bg = (gray >= 245) & (sat <= 18) & (std <= 6.0)
    base = feature & (~soft_bg)

    mid = w // 2
    out: list[BBox] = []
    for hx1, hx2 in [(0, mid), (mid, w)]:
        side = base[:, hx1:hx2]
        if not np.any(side):
            continue
        side_u8 = (side.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(side_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        side_area = float(max(1, h * (hx2 - hx1)))
        min_comp = int(max(1, side_area * cfg.half_rescue_component_area_ratio))
        boxes: list[BBox] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            b = (hx1 + x, y, hx1 + x + bw, y + bh)
            if bbox_area(b) < min_comp:
                continue
            boxes.append(b)
        if not boxes:
            continue
        u = bbox_union(boxes)
        if (bbox_area(u) / float(max(1, h * w))) < cfg.half_rescue_min_side_area_ratio:
            continue
        out.append(clip_bbox(u, w, h))
    return out


def _derive_split_seed_mask(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 3.0)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)

    sat = hsv[:, :, 1]
    edges = cv2.Canny(gray, 70, 150) > 0
    textured = std >= cfg.split_texture_std_threshold
    colorful = sat >= cfg.split_saturation_threshold

    seed = (textured | colorful | edges) & (non_text_mask > 0) & (text_mask == 0)
    seed_mask = (seed.astype(np.uint8) * 255)

    open_k = ratio_to_kernel((h, w), cfg.split_seed_open_ratio, min_size=1)
    close_k = ratio_to_kernel((h, w), cfg.split_seed_close_ratio, min_size=1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return seed_mask


def _extract_seed_bboxes(seed_mask: np.ndarray, cfg: VisualFilterConfig) -> list[BBox]:
    h, w = seed_mask.shape[:2]
    min_area = int(h * w * cfg.split_min_component_area_ratio)
    contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[BBox] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = (x, y, x + bw, y + bh)
        if bbox_area(bbox) < min_area:
            continue
        boxes.append(bbox)
    return boxes


def _grid_split_bboxes(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    rows = max(1, int(cfg.grid_split_rows))
    cols = max(1, int(cfg.grid_split_cols))
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)
    expand = int(round(min(h, w) * max(0.0, cfg.grid_split_expand_ratio)))

    out: list[BBox] = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = w if c == cols - 1 else (c + 1) * cell_w
            y2 = h if r == rows - 1 else (r + 1) * cell_h
            bx = clip_bbox((x1 - expand, y1 - expand, x2 + expand, y2 + expand), w, h)
            gx1, gy1, gx2, gy2 = bx
            g_area = max(1, (gx2 - gx1) * (gy2 - gy1))
            nt_patch = non_text_mask[gy1:gy2, gx1:gx2]
            txt_patch = text_mask[gy1:gy2, gx1:gx2]
            gray_patch = gray[gy1:gy2, gx1:gx2]
            sat_patch = hsv[gy1:gy2, gx1:gx2, 1]
            edges = cv2.Canny(gray_patch, 80, 160) if gray_patch.size else np.zeros((1, 1), dtype=np.uint8)

            nt_ratio = float(np.count_nonzero(nt_patch > 0)) / float(max(1, nt_patch.size))
            txt_ratio = float(np.count_nonzero(txt_patch > 0)) / float(max(1, txt_patch.size))
            edge_density = float(np.count_nonzero(edges > 0)) / float(max(1, edges.size))
            entropy = _entropy(gray_patch)
            sat_mean = float(np.mean(sat_patch)) if sat_patch.size else 0.0

            score = (
                0.40 * min(1.0, edge_density / 0.035)
                + 0.35 * min(1.0, entropy / 5.2)
                + 0.25 * min(1.0, sat_mean / 85.0)
                - 0.30 * min(1.0, txt_ratio / 0.55)
            )
            if nt_ratio < cfg.grid_min_non_text_ratio:
                continue
            if score < cfg.grid_min_visual_score:
                continue
            cell_mask = (nt_patch > 0) & (txt_patch == 0)
            if not np.any(cell_mask):
                continue

            cm = (cell_mask.astype(np.uint8) * 255)
            cm = cv2.morphologyEx(
                cm,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            contours, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_cell_comp_area = int(max(1, g_area * cfg.grid_component_min_cell_area_ratio))
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                if (bw * bh) < min_cell_comp_area:
                    continue
                tx1 = gx1 + x
                ty1 = gy1 + y
                tx2 = gx1 + x + bw
                ty2 = gy1 + y + bh
                tight = clip_bbox((tx1 - expand, ty1 - expand, tx2 + expand, ty2 + expand), w, h)
                wr = (tight[2] - tight[0]) / float(max(1, w))
                hr = (tight[3] - tight[1]) / float(max(1, h))
                if wr > cfg.grid_strip_span_ratio and hr < cfg.grid_strip_other_ratio:
                    continue
                if hr > cfg.grid_strip_span_ratio and wr < cfg.grid_strip_other_ratio:
                    continue
                if bbox_area(tight) / float(max(1, h * w)) < cfg.split_min_component_area_ratio:
                    continue
                out.append(tight)
    return out


def _is_strip_box(bbox: BBox, page_size: tuple[int, int], cfg: VisualFilterConfig) -> bool:
    h, w = page_size
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    return (wr > cfg.strip_span_ratio and hr < cfg.strip_other_ratio) or (
        hr > cfg.strip_span_ratio and wr < cfg.strip_other_ratio
    )


def _recover_column_boxes_from_mask(
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    page_size: tuple[int, int],
) -> list[BBox]:
    h, w = page_size
    if (w / float(max(1, h))) < 1.35:
        return []
    xmid = w // 2
    out: list[BBox] = []
    for x1, x2 in [(0, xmid), (xmid, w)]:
        nt = non_text_mask[:, x1:x2] > 0
        tx = text_mask[:, x1:x2] > 0
        mask = nt & (~tx)
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        bx1 = x1 + int(xs.min())
        bx2 = x1 + int(xs.max()) + 1
        by1 = int(ys.min())
        by2 = int(ys.max()) + 1
        pad = int(round(min(h, w) * 0.01))
        out.append(clip_bbox((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), w, h))
    return out


def _split_strip_box_by_projection(
    bbox: BBox,
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    page_size: tuple[int, int],
    cfg: VisualFilterConfig,
) -> list[BBox]:
    h, w = page_size
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []
    patch = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.size else np.zeros((1, 1), dtype=np.uint8)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV) if patch.size else np.zeros((1, 1, 3), dtype=np.uint8)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.0)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)
    sat = hsv[:, :, 1]
    edges = cv2.Canny(gray, 80, 160) > 0
    visual = (std >= 14.0) | (sat >= 35) | edges
    region = visual & (non_text_mask[y1:y2, x1:x2] > 0) & (text_mask[y1:y2, x1:x2] == 0)
    if not np.any(region):
        return []

    bw = x2 - x1
    bh = y2 - y1
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    min_fill = cfg.strip_projection_min_fill_ratio
    min_run_ratio = cfg.strip_projection_min_run_ratio
    pad = int(round(min(h, w) * cfg.strip_projection_pad_ratio))
    out: list[BBox] = []

    if wr > hr:
        profile = region.mean(axis=0)
        active = profile >= min_fill
        min_len = max(8, int(round(bw * min_run_ratio)))
        start = None
        for i, v in enumerate(active):
            if v and start is None:
                start = i
            elif (not v) and (start is not None):
                if (i - start) >= min_len:
                    sub = region[:, start:i]
                    ys, xs = np.where(sub)
                    if ys.size > 0:
                        bx1 = x1 + start + int(xs.min())
                        bx2 = x1 + start + int(xs.max()) + 1
                        by1 = y1 + int(ys.min())
                        by2 = y1 + int(ys.max()) + 1
                        out.append(clip_bbox((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), w, h))
                start = None
        if start is not None and (len(active) - start) >= min_len:
            sub = region[:, start:len(active)]
            ys, xs = np.where(sub)
            if ys.size > 0:
                bx1 = x1 + start + int(xs.min())
                bx2 = x1 + start + int(xs.max()) + 1
                by1 = y1 + int(ys.min())
                by2 = y1 + int(ys.max()) + 1
                out.append(clip_bbox((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), w, h))
    else:
        profile = region.mean(axis=1)
        active = profile >= min_fill
        min_len = max(8, int(round(bh * min_run_ratio)))
        start = None
        for i, v in enumerate(active):
            if v and start is None:
                start = i
            elif (not v) and (start is not None):
                if (i - start) >= min_len:
                    sub = region[start:i, :]
                    ys, xs = np.where(sub)
                    if ys.size > 0:
                        bx1 = x1 + int(xs.min())
                        bx2 = x1 + int(xs.max()) + 1
                        by1 = y1 + start + int(ys.min())
                        by2 = y1 + start + int(ys.max()) + 1
                        out.append(clip_bbox((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), w, h))
                start = None
        if start is not None and (len(active) - start) >= min_len:
            sub = region[start:len(active), :]
            ys, xs = np.where(sub)
            if ys.size > 0:
                bx1 = x1 + int(xs.min())
                bx2 = x1 + int(xs.max()) + 1
                by1 = y1 + start + int(ys.min())
                by2 = y1 + start + int(ys.max()) + 1
                out.append(clip_bbox((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), w, h))
    return out


def _fallback_visual_component_boxes(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 3.0)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)

    sat_mask = hsv[:, :, 1] >= cfg.fallback_saturation_threshold
    tex_mask = std >= cfg.fallback_texture_std_threshold
    edge_mask = cv2.Canny(gray, 80, 160) > 0
    mask = (sat_mask | tex_mask | edge_mask) & (non_text_mask > 0) & (text_mask == 0)
    fb = (mask.astype(np.uint8) * 255)

    open_k = ratio_to_kernel((h, w), cfg.fallback_edge_open_ratio, min_size=1)
    close_k = ratio_to_kernel((h, w), cfg.fallback_edge_close_ratio, min_size=1)
    fb = cv2.morphologyEx(
        fb,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        iterations=1,
    )
    fb = cv2.morphologyEx(
        fb,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=1,
    )

    min_area = int(max(1, h * w * cfg.fallback_min_component_area_ratio))
    contours, _ = cv2.findContours(fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[BBox] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = clip_bbox((x, y, x + bw, y + bh), w, h)
        if bbox_area(bbox) < min_area:
            continue
        out.append(bbox)
    return out


def _split_large_box_feature_components(
    image_bgr: np.ndarray,
    bbox: BBox,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []
    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.5)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.5)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)
    sat_mask = hsv[:, :, 1] >= cfg.fallback_saturation_threshold
    tex_mask = std >= cfg.fallback_texture_std_threshold
    edge_mask = cv2.Canny(gray, 80, 160) > 0
    nt_patch = non_text_mask[y1:y2, x1:x2] > 0
    tx_patch = text_mask[y1:y2, x1:x2] > 0
    mask = (sat_mask | tex_mask | edge_mask) & nt_patch & (~tx_patch)
    bm = (mask.astype(np.uint8) * 255)
    open_k = ratio_to_kernel((h, w), cfg.large_box_split_open_ratio, min_size=1)
    close_k = ratio_to_kernel((h, w), cfg.large_box_split_close_ratio, min_size=1)
    bm = cv2.morphologyEx(
        bm,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        iterations=1,
    )
    bm = cv2.morphologyEx(
        bm,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=1,
    )
    box_area = max(1, (x2 - x1) * (y2 - y1))
    min_comp = int(max(1, box_area * cfg.large_box_split_min_component_area_ratio))
    def _collect(mask_u8: np.ndarray) -> list[BBox]:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp: list[BBox] = []
        for contour in contours:
            cx, cy, bw, bh = cv2.boundingRect(contour)
            b = (x1 + cx, y1 + cy, x1 + cx + bw, y1 + cy + bh)
            b = clip_bbox(b, w, h)
            if bbox_area(b) < min_comp:
                continue
            tmp.append(b)
        return tmp

    out = _collect(bm)
    box_area_ratio = bbox_area(bbox) / float(max(1, h * w))
    if len(out) >= 2:
        return out

    # Break thin bridges for giant components.
    work = bm.copy()
    for _ in range(3):
        work = cv2.erode(work, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        cand = _collect(work)
        if len(cand) >= 2:
            return cand

    # Projection-based split for stubborn giant regions.
    if box_area_ratio < 0.2:
        return out
    bin_mask = work > 0
    if not np.any(bin_mask):
        return out
    candidates: list[BBox] = []
    col_profile = bin_mask.mean(axis=0)
    row_profile = bin_mask.mean(axis=1)
    col_active = col_profile >= 0.035
    row_active = row_profile >= 0.035
    min_col_run = max(10, int(round((x2 - x1) * 0.14)))
    min_row_run = max(10, int(round((y2 - y1) * 0.14)))

    def _runs(active: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        s = None
        for i, v in enumerate(active):
            if v and s is None:
                s = i
            elif (not v) and (s is not None):
                if (i - s) >= min_len:
                    runs.append((s, i))
                s = None
        if s is not None and (len(active) - s) >= min_len:
            runs.append((s, len(active)))
        return runs

    for c1, c2 in _runs(col_active, min_col_run):
        sub = bin_mask[:, c1:c2]
        ys, xs = np.where(sub)
        if ys.size == 0:
            continue
        b = (x1 + c1 + int(xs.min()), y1 + int(ys.min()), x1 + c1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1)
        b = clip_bbox(b, w, h)
        if bbox_area(b) >= min_comp:
            candidates.append(b)
    for r1, r2 in _runs(row_active, min_row_run):
        sub = bin_mask[r1:r2, :]
        ys, xs = np.where(sub)
        if ys.size == 0:
            continue
        b = (x1 + int(xs.min()), y1 + r1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + r1 + int(ys.max()) + 1)
        b = clip_bbox(b, w, h)
        if bbox_area(b) >= min_comp:
            candidates.append(b)

    if candidates:
        return _merge_overlaps(candidates, 0.12)
    return out


def _page_visual_stats(image_bgr: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges > 0)) / float(max(1, edges.size))
    ent = _entropy(gray)
    white_ratio = float(np.count_nonzero((gray >= 242) & (hsv[:, :, 1] <= 20))) / float(max(1, gray.size))
    return edge_density, ent, white_ratio


def _is_full_page_like_bbox(bbox: BBox, page_size: tuple[int, int]) -> bool:
    h, w = page_size
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    return wr >= 0.97 and hr >= 0.97


def _is_strip_like_bbox(bbox: BBox, page_size: tuple[int, int], cfg: VisualFilterConfig) -> bool:
    h, w = page_size
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    return (wr > cfg.strip_span_ratio and hr < cfg.strip_other_ratio) or (
        hr > cfg.strip_span_ratio and wr < cfg.strip_other_ratio
    )


def _bbox_non_text_fill_ratio(non_text_mask: np.ndarray, bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    patch = non_text_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch > 0)) / float(max(1, patch.size))


def _simple_candidate_score(metrics: CandidateMetrics, page_size: tuple[int, int]) -> float:
    h, w = page_size
    x1, y1, x2, y2 = metrics.bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    area = metrics.area_ratio

    score = (
        2.0 * min(1.0, metrics.edge_density / 0.03)
        + 1.8 * min(1.0, metrics.entropy / 5.2)
        + 0.9 * min(1.0, metrics.fill_ratio / 0.45)
        - 1.2 * min(1.0, metrics.text_overlap_ratio / 0.55)
    )
    if 0.03 <= area <= 0.45:
        score += 0.6
    elif area > 0.62:
        score -= 1.5
    if (wr > 0.92 and hr < 0.33) or (hr > 0.92 and wr < 0.33):
        score -= 1.2
    return float(score)


def _extract_framed_candidate_boxes(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    out: list[BBox] = []
    page_area = float(max(1, h * w))
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        bbox = clip_bbox((x, y, x + bw, y + bh), w, h)
        area_ratio = bbox_area(bbox) / page_area
        if area_ratio < max(0.018, cfg.min_visual_area_ratio * 1.5):
            continue
        if area_ratio > 0.62:
            continue
        short_side_ratio = min((bbox[2] - bbox[0]) / float(max(1, w)), (bbox[3] - bbox[1]) / float(max(1, h)))
        if short_side_ratio < max(0.075, cfg.min_visual_short_side_ratio):
            continue
        if _is_full_page_like_bbox(bbox, (h, w)):
            continue
        if _is_strip_like_bbox(bbox, (h, w), cfg):
            continue
        if _bbox_non_text_fill_ratio(non_text_mask, bbox) < 0.12:
            continue
        x1, y1, x2, y2 = bbox
        t_patch = text_mask[y1:y2, x1:x2]
        t_overlap = float(np.count_nonzero(t_patch > 0)) / float(max(1, t_patch.size))
        if t_overlap > 0.5:
            continue
        out.append(bbox)

    return _merge_overlaps(out, 0.25)


def _keep_simple_candidate(
    metrics: CandidateMetrics,
    bbox: BBox,
    non_text_mask: np.ndarray,
    cfg: VisualFilterConfig,
    page_size: tuple[int, int],
) -> tuple[bool, str]:
    keep, reason = keep_candidate(metrics, cfg, page_size=page_size)
    if not keep:
        return False, reason

    if _is_full_page_like_bbox(bbox, page_size):
        return False, "full_page_like_simple"
    if _is_strip_like_bbox(bbox, page_size, cfg):
        return False, "strip_like_simple"

    if metrics.area_ratio > 0.7:
        return False, "too_large_simple"

    non_text_fill = _bbox_non_text_fill_ratio(non_text_mask, bbox)
    if non_text_fill < 0.1:
        return False, "low_non_text_fill"

    if metrics.text_overlap_ratio > 0.42 and metrics.edge_density < 0.012:
        return False, "text_overlap_simple"

    return True, "ok"


def _extract_colorful_regions(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    page_size: tuple[int, int],
    min_area_ratio: float = 0.002,
) -> list[BBox]:
    """
    Renkli bölgeleri tespit et - çocuk dergilerinde çizimler genellikle canlı renkli.
    Non-text mask içindeki renkli ve texture'lı bölgeleri bulur.
    """
    h, w = page_size
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Canlı renkler (yüksek saturation, orta-yüksek value)
    sat_mask = hsv[:, :, 1] > 45  # Saturation > 45
    val_mask = hsv[:, :, 2] > 60  # Value > 60
    colorful = sat_mask & val_mask

    # Texture analizi ile renksiz ama detaylı bölgeleri de yakala
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 3.0)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)
    textured = std > 12.0

    # Birleştir ve non-text mask ile kesiştir
    visual_hints = (colorful | textured).astype(np.uint8) * 255
    visual_regions = cv2.bitwise_and(visual_hints, non_text_mask)

    # Text mask'i çıkar
    if text_mask is not None:
        visual_regions = cv2.bitwise_and(visual_regions, cv2.bitwise_not(text_mask))

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        visual_regions, connectivity=8
    )

    boxes = []
    min_area = int(h * w * min_area_ratio)

    for i in range(1, num_labels):  # Label 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            boxes.append((x, y, x + bw, y + bh))

    return boxes


def _split_large_visual_into_panels(
    image_bgr: np.ndarray,
    bbox: BBox,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
    page_size: tuple[int, int],
) -> list[BBox]:
    """
    Büyük görsel bloklarını alt panellere böl.
    Renk, texture ve edge bilgilerini kullanarak bölme noktalarını bulur.
    """
    h, w = page_size
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return [bbox]

    area_ratio = bbox_area(bbox) / float(max(1, h * w))

    # Sadece gerçekten büyük blokları böl (> %30 sayfa alanı)
    if area_ratio < 0.30:
        return [bbox]

    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return [bbox]

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Multi-feature map oluştur
    # 1. Edge density
    edges = cv2.Canny(gray, 80, 160) > 0

    # 2. Saturation
    sat = hsv[:, :, 1] > 40

    # 3. Texture (local std)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.5)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.5)
    var = np.maximum(0.0, mean_sq - (mean * mean))
    std = np.sqrt(var)
    textured = std > 14.0

    # Feature birleştir
    feature = (edges | sat | textured).astype(np.uint8) * 255

    # Non-text mask ile kesiştir
    nt_patch = non_text_mask[y1:y2, x1:x2]
    feature = cv2.bitwise_and(feature, nt_patch)

    # Text mask'i çıkar
    if text_mask is not None:
        tx_patch = text_mask[y1:y2, x1:x2]
        feature = cv2.bitwise_and(feature, cv2.bitwise_not(tx_patch))

    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    feature = cv2.morphologyEx(feature, cv2.MORPH_CLOSE, kernel, iterations=1)
    feature = cv2.morphologyEx(feature, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        feature, connectivity=8
    )

    if num_labels <= 1:
        # Fallback: grid-based split
        return _grid_split_single_box(bbox, page_size, rows=2, cols=2)

    # Her component için bbox hesapla
    min_comp_area = int((bw * bh) * 0.02)  # En az %2 component alanı
    panels: list[BBox] = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_comp_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        comp_bw = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Global koordinatlara çevir
        global_x1 = int(x1 + x)
        global_y1 = int(y1 + y)
        global_x2 = int(x1 + x + comp_bw)
        global_y2 = int(y1 + y + comp_bh)

        # Padding ekle
        pad = max(6, int(round(min(h, w) * 0.008)))
        panel = clip_bbox(
            (global_x1 - pad, global_y1 - pad, global_x2 + pad, global_y2 + pad),
            w, h
        )
        panels.append(panel)

    # Eğer hiç panel bulunamadıysa, orijinal box'ı döndür
    if not panels:
        return [bbox]

    # Çok küçük panelleri filtrele
    min_panel_area = int(h * w * 0.004)
    panels = [p for p in panels if bbox_area(p) >= min_panel_area]

    return panels if panels else [bbox]


def _grid_split_single_box(
    bbox: BBox,
    page_size: tuple[int, int],
    rows: int = 2,
    cols: int = 2,
) -> list[BBox]:
    """
    Tek bir büyük kutuyu grid'e böl.
    """
    h, w = page_size
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    cell_w = int(bw // cols)
    cell_h = int(bh // rows)

    panels: list[BBox] = []
    for r in range(rows):
        for c in range(cols):
            cx1 = x1 + c * cell_w
            cy1 = y1 + r * cell_h
            cx2 = x1 + (c + 1) * cell_w if c < cols - 1 else x2
            cy2 = y1 + (r + 1) * cell_h if r < rows - 1 else y2

            pad = max(4, int(round(min(h, w) * 0.005)))
            panel = clip_bbox((cx1 - pad, cy1 - pad, cx2 + pad, cy2 + pad), w, h)
            panels.append(panel)

    return panels


def _extract_visual_candidates_simple(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> tuple[list[CandidateMetrics], list[RejectedCandidate]]:
    h, w = image_bgr.shape[:2]
    contours, _ = cv2.findContours(non_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[CandidateMetrics] = []
    rejected: list[RejectedCandidate] = []
    page_size = (h, w)

    # 0) Renkli bölge tespiti ile ekstra adaylar ekle
    colorful_boxes = _extract_colorful_regions(image_bgr, non_text_mask, text_mask, page_size)

    # 1) Core non-text connected-component candidates.
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = clip_bbox((x, y, x + bw, y + bh), w, h)
        m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        keep, reason = _keep_simple_candidate(m, bbox, non_text_mask, cfg, page_size)
        if keep:
            candidates.append(m)
        else:
            rejected.append(RejectedCandidate(bbox=bbox, reason=reason, area_ratio=m.area_ratio))

    # Renkli bölgeleri de aday olarak ekle
    for bbox in colorful_boxes:
        x1, y1, x2, y2 = bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        # Sadece henüz eklenmemiş olanları ekle
        is_dup = any(iou(bbox, c.bbox) > 0.7 for c in candidates)
        if not is_dup:
            keep, reason = _keep_simple_candidate(m, bbox, non_text_mask, cfg, page_size)
            if keep:
                candidates.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"color_{reason}", area_ratio=m.area_ratio))

    # 2) Guaranteed framed candidate recovery (framed photos/panels should not be missed).
    framed_boxes = _extract_framed_candidate_boxes(image_bgr, non_text_mask, text_mask, cfg)
    for bbox in framed_boxes:
        x1, y1, x2, y2 = bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        keep, reason = _keep_simple_candidate(m, bbox, non_text_mask, cfg, page_size)
        if keep:
            candidates.append(m)
        else:
            rejected.append(RejectedCandidate(bbox=bbox, reason=f"framed_{reason}", area_ratio=m.area_ratio))

    # 3) Merge overlaps and light fragmentation, then re-score.
    boxes = _merge_overlaps([c.bbox for c in candidates], cfg.merge_iou_threshold)
    boxes = _merge_fragmented_boxes(boxes, (h, w), cfg)

    # 3b) BÜYÜK GÖRSELLERİ PANELE BÖL - Çocuk dergileri için kritik!
    split_boxes: list[BBox] = []
    for bbox in boxes:
        area_ratio = bbox_area(bbox) / float(max(1, h * w))
        if area_ratio > 0.25:  # %25'ten büyük görselleri böl
            panels = _split_large_visual_into_panels(
                image_bgr, bbox, non_text_mask, text_mask, cfg, page_size
            )
            split_boxes.extend(panels)
        else:
            split_boxes.append(bbox)

    final: list[CandidateMetrics] = []
    for bbox in split_boxes:
        x1, y1, x2, y2 = bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        keep, reason = _keep_simple_candidate(m, bbox, non_text_mask, cfg, page_size)
        if keep:
            final.append(m)
        else:
            rejected.append(RejectedCandidate(bbox=bbox, reason=f"post_split_{reason}", area_ratio=m.area_ratio))

    # 4) Conservative fallback: choose strongest localized candidates, never full-page.
    if not final:
        ranked: list[CandidateMetrics] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = clip_bbox((x, y, x + bw, y + bh), w, h)
            if _is_full_page_like_bbox(bbox, page_size):
                continue
            if _is_strip_like_bbox(bbox, page_size, cfg):
                continue
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            if (
                m.text_overlap_ratio <= 0.45
                and m.area_ratio >= max(0.012, cfg.min_visual_area_ratio * 0.9)
                and m.area_ratio <= 0.55
                and _bbox_non_text_fill_ratio(non_text_mask, bbox) >= 0.1
                and (m.edge_density >= 0.009 or m.entropy >= 3.2)
            ):
                ranked.append(m)
        ranked.sort(key=lambda x: (x.area_ratio, x.edge_density, x.entropy), reverse=True)
        final.extend(ranked[:2])

    # 5) If still empty, prefer framed detections instead of broad page crops.
    if not final and framed_boxes:
        seeded: list[CandidateMetrics] = []
        for bbox in framed_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            if m.area_ratio <= 0.62:
                seeded.append(m)
        seeded.sort(key=lambda x: (x.area_ratio, x.edge_density, x.entropy), reverse=True)
        final.extend(seeded[:2])

    # 6) Recall rescue: split rejected strip/full-page/large regions into localized panels.
    if len(final) < 2:
        rescue_reasons = {
            "too_canvas_strip",
            "post_merge_too_canvas_strip",
            "strip_like_simple",
            "post_merge_strip_like_simple",
            "too_large_ratio",
            "post_merge_too_large_ratio",
            "too_large_simple",
            "post_merge_too_large_simple",
            "full_page_like_simple",
            "post_merge_full_page_like_simple",
        }
        split_boxes: list[BBox] = []
        for rej in rejected:
            if rej.reason not in rescue_reasons:
                continue
            src = rej.bbox
            parts: list[BBox] = []
            if _is_strip_like_bbox(src, page_size, cfg):
                parts = _split_strip_box_by_projection(src, image_bgr, non_text_mask, text_mask, page_size, cfg)
            if not parts:
                parts = _split_large_box_feature_components(image_bgr, src, non_text_mask, text_mask, cfg)
            split_boxes.extend(parts)

        split_boxes = _merge_overlaps(split_boxes, 0.15)
        split_boxes = _merge_fragmented_boxes(split_boxes, page_size, cfg)
        rescored: list[CandidateMetrics] = []
        for bbox in split_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = _keep_simple_candidate(m, bbox, non_text_mask, cfg, page_size)
            if keep:
                rescored.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"resplit_{reason}", area_ratio=m.area_ratio))
        rescored.sort(key=lambda x: _simple_candidate_score(x, page_size), reverse=True)
        need = max(0, max(2, cfg.simple_mode_max_visuals) - len(final))
        for c in rescored:
            if need <= 0:
                break
            if any(iou(c.bbox, f.bbox) > 0.65 for f in final):
                continue
            final.append(c)
            need -= 1
    # Keep strongest candidates first, then enforce deterministic y/x order.
    dedup: list[CandidateMetrics] = []
    for c in sorted(final, key=lambda x: _simple_candidate_score(x, page_size), reverse=True):
        if any(iou(c.bbox, k.bbox) > 0.75 for k in dedup):
            continue
        dedup.append(c)
    dedup = dedup[: max(1, cfg.simple_mode_max_visuals)]
    final = sorted(dedup, key=lambda c: (c.bbox[1], c.bbox[0]))
    return final, rejected


def extract_visual_candidates(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
    page_archetype: str = "mixed_page",
) -> tuple[list[CandidateMetrics], list[RejectedCandidate]]:
    if cfg.simple_mode:
        return _extract_visual_candidates_simple(image_bgr, non_text_mask, text_mask, cfg)

    contours, _ = cv2.findContours(non_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    h, w = image_bgr.shape[:2]
    text_ratio = float(np.count_nonzero(text_mask > 0)) / float(max(1, text_mask.size))

    # If non-text mask collapses into a giant component, derive extra split seeds
    # from texture/color/edge cues to avoid missing image-heavy regions.
    has_huge_component = False
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if (bw * bh) / float(max(1, h * w)) >= cfg.split_trigger_area_ratio:
            has_huge_component = True
            break
    if has_huge_component or len(contours) <= 1:
        seed_mask = _derive_split_seed_mask(image_bgr, non_text_mask, text_mask, cfg)
        seed_boxes = _extract_seed_bboxes(seed_mask, cfg)
        for x1, y1, x2, y2 in seed_boxes:
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            contours.append(contour)

    # Container-first extras for default mode:
    # add framed/photo-like rectangles and colorful textured regions as additional contours.
    framed_boxes = _extract_framed_candidate_boxes(image_bgr, non_text_mask, text_mask, cfg)
    colorful_boxes = _extract_colorful_regions(
        image_bgr,
        non_text_mask,
        text_mask,
        page_size=(h, w),
        min_area_ratio=max(0.0018, cfg.min_visual_area_ratio * 0.35),
    )
    extra_boxes: list[BBox] = []
    for bbox in framed_boxes + colorful_boxes:
        if any(iou(bbox, e) > 0.75 for e in extra_boxes):
            continue
        extra_boxes.append(bbox)
    for x1, y1, x2, y2 in extra_boxes:
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        contours.append(contour)

    raw_candidates: list[CandidateMetrics] = []
    rejected: list[RejectedCandidate] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = clip_bbox((x, y, x + bw, y + bh), w, h)
        metrics = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        keep, reason = keep_candidate(metrics, cfg, page_size=(h, w))
        if keep:
            raw_candidates.append(metrics)
        else:
            rejected.append(RejectedCandidate(bbox=bbox, reason=reason, area_ratio=metrics.area_ratio))

    merged_boxes = _merge_overlaps([c.bbox for c in raw_candidates], cfg.merge_iou_threshold)
    merged_boxes = _merge_fragmented_boxes(merged_boxes, (h, w), cfg)
    final: list[CandidateMetrics] = []
    for bbox in merged_boxes:
        x1, y1, x2, y2 = bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        metrics = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        keep, reason = keep_candidate(metrics, cfg, page_size=(h, w))
        if keep:
            final.append(metrics)
        else:
            rejected.append(RejectedCandidate(bbox=bbox, reason=f"post_merge_{reason}", area_ratio=metrics.area_ratio))

    if not final and (has_huge_component or len(contours) <= 1):
        # Last-resort recovery for collage-like pages:
        # use grid-derived boxes only when primary extraction produced nothing.
        grid_boxes = _grid_split_bboxes(image_bgr, non_text_mask, text_mask, cfg)
        grid_candidates: list[CandidateMetrics] = []
        for bbox in grid_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = keep_candidate(m, cfg, page_size=(h, w))
            if keep:
                grid_candidates.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"grid_{reason}", area_ratio=m.area_ratio))

        gboxes = _merge_overlaps([c.bbox for c in grid_candidates], cfg.merge_iou_threshold)
        gboxes = _merge_fragmented_boxes(gboxes, (h, w), cfg)
        for bbox in gboxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = keep_candidate(m, cfg, page_size=(h, w))
            if keep:
                final.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"grid_post_{reason}", area_ratio=m.area_ratio))

    # If survivors are mostly strip artifacts on double-page layouts, replace with
    # safer left/right page-level visual boxes.
    # Split strip-like survivors into localized regions before deciding fallback replacement.
    if final:
        split_candidates: list[CandidateMetrics] = []
        split_applied = False
        for c in final:
            if not _is_strip_box(c.bbox, (h, w), cfg):
                split_candidates.append(c)
                continue
            pieces = _split_strip_box_by_projection(c.bbox, image_bgr, non_text_mask, text_mask, (h, w), cfg)
            if not pieces:
                split_candidates.append(c)
                continue
            split_applied = True
            for bbox in pieces:
                x1, y1, x2, y2 = bbox
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
                keep, reason = keep_candidate(m, cfg, page_size=(h, w))
                if keep:
                    split_candidates.append(m)
                else:
                    rejected.append(RejectedCandidate(bbox=bbox, reason=f"strip_split_{reason}", area_ratio=m.area_ratio))
        if split_applied:
            dedup = _merge_overlaps([c.bbox for c in split_candidates], cfg.merge_iou_threshold)
            final = []
            for bbox in dedup:
                x1, y1, x2, y2 = bbox
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
                keep, reason = keep_candidate(m, cfg, page_size=(h, w))
                if keep:
                    final.append(m)
                else:
                    rejected.append(RejectedCandidate(bbox=bbox, reason=f"strip_split_post_{reason}", area_ratio=m.area_ratio))

    if final:
        strip_count = sum(1 for c in final if _is_strip_box(c.bbox, (h, w), cfg))
        if strip_count >= max(2, int(0.6 * len(final))):
            recovered = _recover_column_boxes_from_mask(non_text_mask, text_mask, (h, w))
            if recovered:
                final = []
                for bbox in recovered:
                    x1, y1, x2, y2 = bbox
                    contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                    m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
                    final.append(m)

    # Last fallback: derive local components from texture/color/edge map, but only
    # when no candidate survived, to avoid aggressive behavior on already-good pages.
    if not final:
        fb_boxes = _fallback_visual_component_boxes(image_bgr, non_text_mask, text_mask, cfg)
        fb_boxes = _merge_overlaps(fb_boxes, cfg.merge_iou_threshold)
        fb_boxes = _merge_fragmented_boxes(fb_boxes, (h, w), cfg)
        for bbox in fb_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = keep_candidate(m, cfg, page_size=(h, w))
            if keep:
                final.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"fallback_{reason}", area_ratio=m.area_ratio))

    # Controlled rescue: split rejected large/strip boxes into smaller feature-rich parts.
    if not final:
        rescue_sources = [
            r.bbox
            for r in rejected
            if ("too_large_ratio" in r.reason or "too_canvas_strip" in r.reason) and r.area_ratio >= 0.22
        ]
        split_boxes: list[BBox] = []
        for src in rescue_sources:
            split_boxes.extend(_split_large_box_feature_components(image_bgr, src, non_text_mask, text_mask, cfg))
        split_boxes = _merge_overlaps(split_boxes, cfg.merge_iou_threshold)
        split_boxes = _merge_fragmented_boxes(split_boxes, (h, w), cfg)
        for bbox in split_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = keep_candidate(m, cfg, page_size=(h, w))
            if keep:
                final.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"resplit_{reason}", area_ratio=m.area_ratio))

    # If we still have no candidates, strip-like rejects can be split into local pieces.
    if not final:
        strip_sources = [r.bbox for r in rejected if "too_canvas_strip" in r.reason and r.area_ratio >= 0.15]
        split_boxes: list[BBox] = []
        for src in strip_sources:
            split_boxes.extend(_split_strip_box_by_projection(src, image_bgr, non_text_mask, text_mask, (h, w), cfg))
        split_boxes = _merge_overlaps(split_boxes, 0.12)
        split_boxes = _merge_fragmented_boxes(split_boxes, (h, w), cfg)
        for bbox in split_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = keep_candidate(m, cfg, page_size=(h, w))
            if keep:
                final.append(m)
            else:
                rejected.append(RejectedCandidate(bbox=bbox, reason=f"strip_recover_{reason}", area_ratio=m.area_ratio))

    # Recall boost: if page ended with too few visuals, try extracting extra pieces
    # from rejected strip regions (without replacing existing survivors).
    if len(final) < 2:
        strip_sources = [r.bbox for r in rejected if "too_canvas_strip" in r.reason and r.area_ratio >= 0.12]
        extra_boxes: list[BBox] = []
        for src in strip_sources:
            extra_boxes.extend(_split_strip_box_by_projection(src, image_bgr, non_text_mask, text_mask, (h, w), cfg))
        extra_boxes = _merge_overlaps(extra_boxes, 0.12)
        for bbox in extra_boxes:
            if any(iou(bbox, c.bbox) > 0.7 for c in final):
                continue
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            keep, reason = keep_candidate(m, cfg, page_size=(h, w))
            if keep:
                final.append(m)
            elif reason == "too_small_ratio" and m.area_ratio >= 0.0038 and m.entropy >= 3.0 and m.edge_density >= 0.01:
                final.append(m)

    # Last resort with bounded width: keep heavily constrained, only for true photo pages.
    if not final:
        page_edge, page_entropy, page_white = _page_visual_stats(image_bgr)
        if (
            page_archetype == "full_photo_page"
            and
            page_edge >= cfg.fullpage_rescue_min_edge_density
            and page_entropy >= cfg.fullpage_rescue_min_entropy
            and page_white <= cfg.fullpage_rescue_max_white_ratio
            and text_ratio <= cfg.fullpage_rescue_max_text_ratio
        ):
            mid = w // 2
            pad = max(8, int(round(min(h, w) * 0.03)))
            seed_boxes = [
                clip_bbox((pad, pad, mid - pad, h - pad), w, h),
                clip_bbox((mid + pad, pad, w - pad, h - pad), w, h),
            ]
            added = 0
            for bbox in seed_boxes:
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
                if m.edge_density >= 0.01 and m.entropy >= 3.2 and m.text_overlap_ratio <= 0.5:
                    final.append(m)
                    added += 1
            if added == 0:
                bbox = clip_bbox((pad, pad, w - pad, h - pad), w, h)
                x1, y1, x2, y2 = bbox
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
                # Do not emit broad canvas-like bands as fallback.
                if not _is_full_width_band(bbox, (h, w)):
                    final.append(m)

    # Wide-spread rescue: only for true full-photo pages to avoid page-halves on editorial layouts.
    if (w / float(max(1, h))) >= 1.35:
        trigger = False
        if not final:
            trigger = page_archetype == "full_photo_page"
        elif len(final) == 1:
            b = final[0].bbox
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            wr = bw / float(max(1, w))
            hr = bh / float(max(1, h))
            ar = bbox_area(b) / float(max(1, h * w))
            cx = (b[0] + b[2]) / 2.0
            side_locked = (cx < 0.42 * w) or (cx > 0.58 * w)
            if page_archetype == "full_photo_page" and ((wr > 0.7 and hr < 0.35) or (ar < 0.1) or side_locked):
                trigger = True
        if trigger:
            half_boxes = _extract_wide_half_panels(image_bgr, non_text_mask, text_mask, cfg, (h, w))
            if half_boxes:
                final = []
                for bbox in half_boxes:
                    if _is_full_width_band(bbox, (h, w)):
                        continue
                    x1, y1, x2, y2 = bbox
                    contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                    final.append(_compute_metrics(image_bgr, contour, bbox, text_mask=text_mask))

    # Split over-wide giant boxes only for full-photo pages.
    if final and page_archetype == "full_photo_page":
        split_final: list[CandidateMetrics] = []
        for c in final:
            parts = _split_huge_wide_box(c.bbox, image_bgr, non_text_mask, text_mask, (h, w))
            for bbox in parts:
                if _is_full_width_band(bbox, (h, w)):
                    continue
                x1, y1, x2, y2 = bbox
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                split_final.append(_compute_metrics(image_bgr, contour, bbox, text_mask=text_mask))
        if split_final:
            final = split_final

    # Final guardrail: drop residual page-wide bands unless this is truly a photo page.
    if final and page_archetype != "full_photo_page":
        final = [c for c in final if not _is_full_width_band(c.bbox, (h, w))]

    # Final panel-level consolidation to avoid fragmented crops.
    coalesced = _coalesce_visual_panels(final, image_bgr, text_mask, cfg, (h, w))
    if coalesced:
        final = coalesced

    # Apply the same guardrail after coalescing as well; merging can re-create wide canvas bands.
    if final and page_archetype != "full_photo_page":
        final = [c for c in final if not _is_full_width_band(c.bbox, (h, w))]
        if (not final) and framed_boxes:
            # If everything was a band, fall back to framed candidates.
            seeded: list[CandidateMetrics] = []
            for bbox in framed_boxes:
                x1, y1, x2, y2 = bbox
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
                keep, _ = keep_candidate(m, cfg, page_size=(h, w))
                if keep and not _is_full_width_band(m.bbox, (h, w)):
                    seeded.append(m)
            if seeded:
                final = seeded

    # If all candidates collapsed after guardrails, recover a center-focused visual box.
    if not final and page_archetype != "full_photo_page":
        focus_boxes = _extract_center_focus_boxes(image_bgr, text_mask, (h, w))
        for bbox in focus_boxes:
            x1, y1, x2, y2 = bbox
            contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
            m = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
            if not _is_full_width_band(bbox, (h, w)):
                final.append(m)

    final.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return final, rejected


def _classify_visual(metrics: CandidateMetrics, page_size: tuple[int, int]) -> VisualClass:
    h, w = page_size
    x1, y1, x2, y2 = metrics.bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    if metrics.area_ratio >= 0.45:
        return "full_compositions"
    if metrics.fill_ratio >= 0.43 and 0.02 <= metrics.area_ratio <= 0.2 and metrics.aspect_ratio <= 2.2:
        return "collage_cards"
    if metrics.fill_ratio >= 0.35 and metrics.aspect_ratio <= 4.0 and metrics.text_overlap_ratio <= 0.6:
        if metrics.text_overlap_ratio >= 0.1 and wr >= 0.22 and hr >= 0.12:
            return "comic_panels"
        return "framed_rectangular"
    if metrics.edge_density >= 0.012 or metrics.entropy >= 3.8:
        return "freeform_illustrations"
    return "ambiguous_review"


def _tighten_to_feature(
    bbox: BBox,
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    feature_mode: str,
    pad_ratio: float,
) -> BBox:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return bbox
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.0)
    std = np.sqrt(np.maximum(0.0, mean_sq - (mean * mean)))
    sat = hsv[:, :, 1]
    edges = cv2.Canny(gray, 80, 160) > 0
    nt = non_text_mask[y1:y2, x1:x2] > 0
    tx = text_mask[y1:y2, x1:x2] > 0
    if feature_mode == "framed":
        feat = nt & (~tx)
    else:
        feat = (edges | (std >= 12.0) | (sat >= 30)) & nt
    if not np.any(feat):
        return bbox
    ys, xs = np.where(feat)
    bx1 = x1 + int(xs.min())
    bx2 = x1 + int(xs.max()) + 1
    by1 = y1 + int(ys.min())
    by2 = y1 + int(ys.max()) + 1
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    # Content-aware breathing room: a bit more bottom room for freeform.
    if feature_mode == "freeform":
        pl = int(round(bw * pad_ratio))
        pr = int(round(bw * pad_ratio))
        pt = int(round(bh * (pad_ratio * 0.9)))
        pb = int(round(bh * (pad_ratio * 1.3)))
    else:
        pl = pr = pt = pb = int(round(min(bw, bh) * pad_ratio))
    return clip_bbox((bx1 - pl, by1 - pt, bx2 + pr, by2 + pb), w, h)


def _qa_decision(
    metrics: CandidateMetrics,
    page_size: tuple[int, int],
    visual_class: VisualClass,
) -> tuple[bool, list[str]]:
    h, w = page_size
    x1, y1, x2, y2 = metrics.bbox
    reasons: list[str] = []
    if visual_class != "full_compositions":
        if x1 <= 1 or y1 <= 1 or x2 >= (w - 1) or y2 >= (h - 1):
            reasons.append("touches_page_edge")
    if metrics.text_overlap_ratio > 0.6 and metrics.edge_density < 0.02:
        reasons.append("text_heavy")
    if metrics.area_ratio < 0.01 and metrics.entropy < 3.2:
        reasons.append("tiny_or_weak")
    return (len(reasons) > 0), reasons


def _decision_quality_score(d: VisualDecision, page_size: tuple[int, int]) -> float:
    h, w = page_size
    m = d.metrics
    x1, y1, x2, y2 = m.bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))

    base = (
        2.0 * min(1.0, m.edge_density / 0.03)
        + 1.8 * min(1.0, m.entropy / 5.2)
        + 0.7 * min(1.0, m.fill_ratio / 0.45)
        - 1.6 * min(1.0, m.text_overlap_ratio / 0.55)
    )
    # Prefer content-sized boxes over giant page-wide boxes unless unavoidable.
    if m.area_ratio > 0.72:
        base -= 2.4
    elif m.area_ratio > 0.55:
        base -= 1.3
    elif 0.02 <= m.area_ratio <= 0.5:
        base += 0.55
    # Penalize canvas-like strips.
    if (wr > 0.92 and hr < 0.33) or (hr > 0.92 and wr < 0.33):
        base -= 1.8
    if d.needs_review:
        base -= 0.6
    return float(base)


def _expand_large_decision_into_subdecisions(
    d: VisualDecision,
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[VisualDecision]:
    h, w = image_bgr.shape[:2]
    m = d.metrics
    # Only try to split very large compositions.
    if m.area_ratio < 0.58:
        return [d]
    sub_boxes = _split_large_box_feature_components(image_bgr, m.bbox, non_text_mask, text_mask, cfg)
    if len(sub_boxes) < 2:
        return [d]

    out: list[VisualDecision] = []
    for bbox in _merge_overlaps(sub_boxes, 0.15):
        x1, y1, x2, y2 = bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        nm = _compute_metrics(image_bgr, contour, bbox, text_mask=text_mask)
        keep, _ = keep_candidate(nm, cfg, page_size=(h, w))
        if not keep:
            continue
        ncls = _classify_visual(nm, (h, w))
        nreview, nreasons = _qa_decision(nm, (h, w), ncls)
        out.append(
            VisualDecision(
                metrics=nm,
                visual_class=ncls if not nreview else "ambiguous_review",
                needs_review=nreview,
                review_reasons=nreasons,
            )
        )
    if len(out) < 2:
        return [d]

    old_score = _decision_quality_score(d, (h, w))
    new_score = sum(_decision_quality_score(x, (h, w)) for x in out)
    # Replace giant single crop only when split is clearly better.
    if new_score >= (old_score + 0.8):
        return out
    return [d]


def _select_decision_set(decisions: list[VisualDecision], page_size: tuple[int, int]) -> list[VisualDecision]:
    if not decisions:
        return []
    h, w = page_size
    ranked = sorted(
        decisions,
        key=lambda d: _decision_quality_score(d, (h, w)),
        reverse=True,
    )

    selected: list[VisualDecision] = []
    for d in ranked:
        # Keep strong non-overlapping candidates first.
        if any(iou(d.metrics.bbox, s.metrics.bbox) > 0.72 for s in selected):
            continue
        selected.append(d)

    # Prevent over-collapse to one giant crop when solid alternatives exist.
    if len(selected) == 1:
        only = selected[0]
        if only.metrics.area_ratio > 0.7:
            alt = [
                d
                for d in ranked[1:]
                if d.metrics.area_ratio >= 0.05 and iou(d.metrics.bbox, only.metrics.bbox) < 0.55
            ]
            if alt:
                selected.extend(alt[:2])

    # Avoid over-fragmentation: keep strongest subset when many tiny crops survive.
    if len(selected) > 5:
        small_count = sum(1 for d in selected if d.metrics.area_ratio <= 0.08)
        coverage = sum(d.metrics.area_ratio for d in selected)
        if small_count >= 4 and coverage <= 0.7:
            ranked_selected = sorted(selected, key=lambda d: _decision_quality_score(d, (h, w)), reverse=True)
            selected = ranked_selected[:5]

    selected.sort(key=lambda x: (x.metrics.bbox[1], x.metrics.bbox[0]))
    return selected


def _should_merge_parent_child(
    a: VisualDecision,
    b: VisualDecision,
    page_size: tuple[int, int],
    cfg: VisualFilterConfig,
) -> bool:
    h, w = page_size
    pa = a.metrics.bbox
    pb = b.metrics.bbox
    inter = _intersection_area(pa, pb)
    if inter <= 0:
        return False
    min_a = float(max(1, min(bbox_area(pa), bbox_area(pb))))
    ov = inter / min_a
    if ov < cfg.parent_child_iou_merge:
        return False
    ua = bbox_area(bbox_union([pa, pb])) / float(max(1, h * w))
    if ua > cfg.max_visual_area_ratio:
        return False
    return True


def _merge_parent_child_decisions(
    decisions: list[VisualDecision],
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[VisualDecision]:
    if len(decisions) < 2:
        return decisions
    h, w = image_bgr.shape[:2]
    cur = list(decisions)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        used = [False] * len(cur)
        nxt: list[VisualDecision] = []
        for i in range(len(cur)):
            if used[i]:
                continue
            used[i] = True
            current = cur[i]
            for j in range(i + 1, len(cur)):
                if used[j]:
                    continue
                other = cur[j]
                if not _should_merge_parent_child(current, other, (h, w), cfg):
                    continue
                ub = bbox_union([current.metrics.bbox, other.metrics.bbox])
                x1, y1, x2, y2 = ub
                contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                m = _compute_metrics(image_bgr, contour, ub, text_mask=text_mask)
                cls = _classify_visual(m, (h, w))
                rv, rs = _qa_decision(m, (h, w), cls)
                current = VisualDecision(
                    metrics=m,
                    visual_class=cls if not rv else "ambiguous_review",
                    needs_review=rv,
                    review_reasons=rs,
                )
                used[j] = True
                changed = True
            nxt.append(current)
        cur = nxt
    cur.sort(key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]))
    return cur


def _route_decisions_by_page_type(
    decisions: list[VisualDecision],
    page_archetype: str,
    page_size: tuple[int, int],
    cfg: VisualFilterConfig,
) -> list[VisualDecision]:
    if not decisions:
        return []
    h, w = page_size
    ranked = sorted(decisions, key=lambda d: _decision_quality_score(d, (h, w)), reverse=True)

    if page_archetype == "full_photo_page":
        filtered = [d for d in ranked if d.metrics.area_ratio >= 0.04]
        return sorted(filtered[: cfg.max_visuals_full_photo_page], key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]))
    if page_archetype == "collage_card_page":
        filtered = [d for d in ranked if d.metrics.area_ratio >= 0.02]
        return sorted(
            filtered[: cfg.max_visuals_collage_card_page],
            key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]),
        )
    if page_archetype == "comic_panel_page":
        return sorted(
            ranked[: cfg.max_visuals_comic_panel_page],
            key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]),
        )
    if page_archetype == "text_heavy_editorial_page":
        filtered = [d for d in ranked if d.metrics.area_ratio >= 0.03]
        return sorted(
            filtered[: cfg.max_visuals_text_heavy_editorial_page],
            key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]),
        )
    return sorted(ranked, key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]))


def _build_visual_decisions_simple(
    candidates: list[CandidateMetrics],
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> list[VisualDecision]:
    h, w = image_bgr.shape[:2]
    decisions: list[VisualDecision] = []
    for c in candidates:
        cls = _classify_visual(c, (h, w))
        refined = c.bbox
        if cls in {"framed_rectangular", "comic_panels", "collage_cards"}:
            refined = _tighten_to_feature(c.bbox, image_bgr, non_text_mask, text_mask, "framed", pad_ratio=0.018)
        elif cls == "freeform_illustrations":
            refined = _tighten_to_feature(c.bbox, image_bgr, non_text_mask, text_mask, "freeform", pad_ratio=0.04)
        x1, y1, x2, y2 = refined
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        m = _compute_metrics(image_bgr, contour, refined, text_mask=text_mask)
        needs_review, reasons = _qa_decision(m, (h, w), cls)
        decisions.append(
            VisualDecision(
                metrics=m,
                visual_class=cls if not needs_review else "ambiguous_review",
                needs_review=needs_review,
                review_reasons=reasons,
            )
        )

    dedup: list[VisualDecision] = []
    for d in sorted(decisions, key=lambda x: _decision_quality_score(x, (h, w)), reverse=True):
        if any(iou(d.metrics.bbox, k.metrics.bbox) > 0.75 for k in dedup):
            continue
        dedup.append(d)
    dedup = sorted(dedup[: max(1, cfg.simple_mode_max_visuals)], key=lambda d: (d.metrics.bbox[1], d.metrics.bbox[0]))
    return dedup


def build_visual_decisions(
    candidates: list[CandidateMetrics],
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
    page_archetype: str = "mixed_page",
) -> list[VisualDecision]:
    if cfg.simple_mode:
        return _build_visual_decisions_simple(candidates, image_bgr, non_text_mask, text_mask, cfg)

    h, w = image_bgr.shape[:2]
    # Safety net: if extraction returned only one side on a wide spread,
    # force a second pass half-panel recovery before classification.
    if len(candidates) == 1 and (w / float(max(1, h))) >= 1.35:
        b = candidates[0].bbox
        cx = (b[0] + b[2]) / 2.0
        side_locked = (cx < 0.42 * w) or (cx > 0.58 * w)
        if side_locked:
            half_boxes = _extract_wide_half_panels(image_bgr, non_text_mask, text_mask, cfg, (h, w))
            if len(half_boxes) >= 2:
                tmp: list[CandidateMetrics] = []
                for bbox in half_boxes:
                    x1, y1, x2, y2 = bbox
                    contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                    tmp.append(_compute_metrics(image_bgr, contour, bbox, text_mask=text_mask))
                candidates = tmp

    decisions: list[VisualDecision] = []
    for c in candidates:
        cls = _classify_visual(c, (h, w))
        refined_bbox = c.bbox
        if cls in {"framed_rectangular", "comic_panels", "collage_cards"}:
            refined_bbox = _tighten_to_feature(c.bbox, image_bgr, non_text_mask, text_mask, "framed", pad_ratio=0.02)
        elif cls == "freeform_illustrations":
            refined_bbox = _tighten_to_feature(c.bbox, image_bgr, non_text_mask, text_mask, "freeform", pad_ratio=0.05)
        x1, y1, x2, y2 = refined_bbox
        contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        m = _compute_metrics(image_bgr, contour, refined_bbox, text_mask=text_mask)
        needs_review, reasons = _qa_decision(m, (h, w), cls)
        out_class: VisualClass = cls if not needs_review else "ambiguous_review"
        decisions.append(VisualDecision(metrics=m, visual_class=out_class, needs_review=needs_review, review_reasons=reasons))

    expanded: list[VisualDecision] = []
    for d in decisions:
        expanded.extend(_expand_large_decision_into_subdecisions(d, image_bgr, non_text_mask, text_mask, cfg))

    # Drop near-duplicates after refinement.
    dedup: list[VisualDecision] = []
    for d in sorted(expanded, key=lambda x: (x.metrics.bbox[1], x.metrics.bbox[0])):
        if any(iou(d.metrics.bbox, k.metrics.bbox) > 0.75 for k in dedup):
            continue
        dedup.append(d)

    final = _select_decision_set(dedup, (h, w))
    final = _merge_parent_child_decisions(final, image_bgr, text_mask, cfg)
    final = _route_decisions_by_page_type(final, page_archetype, (h, w), cfg)
    return final
