from __future__ import annotations

from dataclasses import dataclass

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
    edge_touches = _touches_edges(metrics.bbox, w, h)
    if edge_touches >= 3 and metrics.area_ratio > cfg.edge_touch_reject_area_ratio:
        return False, "edge_touch_large"
    if metrics.text_overlap_ratio > cfg.max_text_overlap_ratio:
        return False, "text_heavy_region"
    if metrics.edge_density < cfg.min_edge_density:
        return False, "low_edge_density"
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


def extract_visual_candidates(
    image_bgr: np.ndarray,
    non_text_mask: np.ndarray,
    text_mask: np.ndarray,
    cfg: VisualFilterConfig,
) -> tuple[list[CandidateMetrics], list[RejectedCandidate]]:
    contours, _ = cv2.findContours(non_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    h, w = image_bgr.shape[:2]

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

    final.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return final, rejected
