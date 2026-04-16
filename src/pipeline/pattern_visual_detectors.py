from __future__ import annotations

import cv2
import numpy as np

from src.utils.geometry import BBox, bbox_area, bbox_union, clip_bbox, iou


def _text_overlap_ratio(text_mask: np.ndarray, bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    patch = text_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch > 0)) / float(max(1, patch.size))


def _visual_fill_ratio(visual_mask: np.ndarray, bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    patch = visual_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch > 0)) / float(max(1, patch.size))


def _is_band_like(bbox: BBox, page_size: tuple[int, int]) -> bool:
    h, w = page_size
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    wr = bw / float(max(1, w))
    hr = bh / float(max(1, h))
    ar = (bw * bh) / float(max(1, w * h))
    return ((wr >= 0.92 and hr <= 0.42) or (hr >= 0.92 and wr <= 0.42)) and ar >= 0.15


def _merge_boxes(boxes: list[BBox], iou_thr: float = 0.22) -> list[BBox]:
    if not boxes:
        return []
    cur = list(boxes)
    changed = True
    while changed:
        changed = False
        used = [False] * len(cur)
        merged: list[BBox] = []
        for i in range(len(cur)):
            if used[i]:
                continue
            used[i] = True
            b = cur[i]
            for j in range(i + 1, len(cur)):
                if used[j]:
                    continue
                if iou(b, cur[j]) >= iou_thr:
                    b = bbox_union([b, cur[j]])
                    used[j] = True
                    changed = True
            merged.append(b)
        cur = merged
    return cur


def _drop_contained(boxes: list[BBox]) -> list[BBox]:
    out: list[BBox] = []
    for i, a in enumerate(boxes):
        keep = True
        aa = float(max(1, bbox_area(a)))
        for j, b in enumerate(boxes):
            if i == j:
                continue
            inter_x1 = max(a[0], b[0])
            inter_y1 = max(a[1], b[1])
            inter_x2 = min(a[2], b[2])
            inter_y2 = min(a[3], b[3])
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            if (inter / aa) >= 0.92 and bbox_area(b) >= bbox_area(a):
                keep = False
                break
        if keep:
            out.append(a)
    return out


def _side_coverage_ratio(bbox: BBox, page_size: tuple[int, int]) -> float:
    h, w = page_size
    half_w = max(1.0, w / 2.0)
    half_area = half_w * float(max(1, h))
    return bbox_area(bbox) / float(max(1.0, half_area))


def _rect_panel_candidates(image_bgr: np.ndarray, text_mask: np.ndarray, visual_mask: np.ndarray) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 70, 150)
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[BBox] = []
    page_area = float(max(1, h * w))
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = clip_bbox((x, y, x + bw, y + bh), w, h)
        ar = bbox_area(b) / page_area
        if ar < 0.03 or ar > 0.72:
            continue
        if _is_band_like(b, (h, w)):
            continue
        aspect = max(bw / float(max(1, bh)), bh / float(max(1, bw)))
        if aspect > 4.2:
            continue
        fill = cv2.contourArea(c) / float(max(1, bw * bh))
        if fill < 0.18 and ar < 0.16:
            continue
        tovr = _text_overlap_ratio(text_mask, b)
        if tovr > 0.62:
            continue
        vfill = _visual_fill_ratio(visual_mask, b)
        if vfill < 0.08:
            continue
        out.append(b)
    return out


def _component_candidates(visual_mask: np.ndarray, text_mask: np.ndarray, page_size: tuple[int, int]) -> list[BBox]:
    h, w = page_size
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(visual_mask, connectivity=8)
    page_area = float(max(1, h * w))
    out: list[BBox] = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        b = clip_bbox((x, y, x + bw, y + bh), w, h)
        ar = bbox_area(b) / page_area
        if ar < 0.04 or ar > 0.72:
            continue
        if _is_band_like(b, (h, w)):
            continue
        if _text_overlap_ratio(text_mask, b) > 0.55:
            continue
        out.append(b)
    return out


def _spread_half_fallback(image_bgr: np.ndarray, text_mask: np.ndarray) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    if (w / float(max(1, h))) < 1.35:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    non_white = ((gray < 243) | (hsv[:, :, 1] > 18)).astype(np.uint8)
    mid = w // 2
    halves = [(0, 0, mid, h), (mid, 0, w, h)]
    out: list[BBox] = []
    for b in halves:
        x1, y1, x2, y2 = b
        patch = non_white[y1:y2, x1:x2]
        ratio = float(np.count_nonzero(patch > 0)) / float(max(1, patch.size))
        tovr = _text_overlap_ratio(text_mask, b)
        if ratio >= 0.18 and tovr <= 0.58:
            out.append(b)
    return out


def detect_pattern_visual_boxes(image_bgr: np.ndarray, text_mask: np.ndarray, prefer_two: bool = True) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.5)
    mean_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 2.5)
    std = np.sqrt(np.maximum(0.0, mean_sq - (mean * mean)))
    edges = cv2.Canny(gray, 80, 160) > 0
    sat = hsv[:, :, 1] > 26

    text_halo = cv2.dilate(
        (text_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    visual_seed = ((edges | sat | (std > 10.0)) & (~text_halo)).astype(np.uint8) * 255
    visual_seed = cv2.morphologyEx(
        visual_seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    boxes = _rect_panel_candidates(image_bgr, text_mask, visual_seed)
    boxes.extend(_component_candidates(visual_seed, text_mask, (h, w)))
    boxes = _merge_boxes(boxes, iou_thr=0.24)
    boxes = _drop_contained(boxes)
    boxes = [b for b in boxes if not _is_band_like(b, (h, w))]

    if not boxes:
        boxes = _spread_half_fallback(image_bgr, text_mask)

    # Rank by area, then prefer lower text overlap and higher visual fill.
    scored: list[tuple[float, BBox]] = []
    for b in boxes:
        ar = bbox_area(b) / float(max(1, h * w))
        tovr = _text_overlap_ratio(text_mask, b)
        vfill = _visual_fill_ratio(visual_seed, b)
        score = (2.6 * ar) + (1.0 * vfill) - (1.6 * tovr)
        scored.append((score, b))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[BBox] = []
    max_keep = 2 if prefer_two else 4
    min_area_keep = 0.06

    # On spread pages, prefer side-balanced picks first.
    if (w / float(max(1, h))) >= 1.35 and prefer_two:
        left_best: tuple[float, BBox] | None = None
        right_best: tuple[float, BBox] | None = None
        for s, b in scored:
            ar = bbox_area(b) / float(max(1, h * w))
            if ar < min_area_keep:
                continue
            side_cov = _side_coverage_ratio(b, (h, w))
            if side_cov < 0.22:
                continue
            cx = (b[0] + b[2]) / 2.0
            if cx <= (w / 2.0):
                if left_best is None or s > left_best[0]:
                    left_best = (s, b)
            else:
                if right_best is None or s > right_best[0]:
                    right_best = (s, b)
        if left_best is not None:
            out.append(left_best[1])
        if right_best is not None and (not out or iou(right_best[1], out[0]) <= 0.68):
            out.append(right_best[1])

    for _, b in scored:
        if len(out) >= max_keep:
            break
        ar = bbox_area(b) / float(max(1, h * w))
        if ar < min_area_keep:
            continue
        if (w / float(max(1, h))) >= 1.35 and _side_coverage_ratio(b, (h, w)) < 0.20:
            continue
        if any(iou(b, k) > 0.68 for k in out):
            continue
        out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))


def detect_stacked_framed_cards(
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    side: str = "left",
) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 70, 150)
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    side_mid = w // 2
    candidates: list[BBox] = []
    page_area = float(max(1, h * w))
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = clip_bbox((x, y, x + bw, y + bh), w, h)
        ar = bbox_area(b) / page_area
        if ar < 0.025 or ar > 0.25:
            continue
        if _is_band_like(b, (h, w)):
            continue
        cx = (b[0] + b[2]) / 2.0
        if side == "left" and cx > side_mid:
            continue
        if side == "right" and cx <= side_mid:
            continue
        tovr = _text_overlap_ratio(text_mask, b)
        if tovr > 0.38:
            continue
        aspect = max(bw / float(max(1, bh)), bh / float(max(1, bw)))
        if aspect > 3.4:
            continue
        candidates.append(b)
    candidates = _merge_boxes(candidates, iou_thr=0.28)
    candidates = _drop_contained(candidates)
    if len(candidates) < 2:
        return []
    candidates = sorted(candidates, key=lambda bb: (bb[1], bb[0]))
    # Find top-bottom pair with x-alignment.
    best: list[BBox] = []
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if b[1] <= a[1]:
                continue
            axc = (a[0] + a[2]) / 2.0
            bxc = (b[0] + b[2]) / 2.0
            x_align = 1.0 - min(1.0, abs(axc - bxc) / float(max(1, w * 0.25)))
            dy = (b[1] - a[1]) / float(max(1, h))
            if dy < 0.16:
                continue
            score = x_align + min(0.9, dy)
            if score > best_score:
                best_score = score
                best = [a, b]
    return best


def detect_puzzle_quadrant_boxes(image_bgr: np.ndarray, text_mask: np.ndarray) -> list[BBox]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160) > 0
    non_white = (gray < 243) | (hsv[:, :, 1] > 18)
    rows = [(0, h // 2), (h // 2, h)]
    cols = [(0, w // 2), (w // 2, w)]
    out: list[BBox] = []
    for y1, y2 in rows:
        for x1, x2 in cols:
            b = (x1, y1, x2, y2)
            p_edges = edges[y1:y2, x1:x2]
            p_nonw = non_white[y1:y2, x1:x2]
            edge_ratio = float(np.count_nonzero(p_edges)) / float(max(1, p_edges.size))
            nonw_ratio = float(np.count_nonzero(p_nonw)) / float(max(1, p_nonw.size))
            tovr = _text_overlap_ratio(text_mask, b)
            if edge_ratio >= 0.018 and nonw_ratio >= 0.16 and tovr <= 0.72:
                out.append(b)
    out = _merge_boxes(out, iou_thr=0.15)
    return sorted(out, key=lambda bb: (bb[1], bb[0]))
