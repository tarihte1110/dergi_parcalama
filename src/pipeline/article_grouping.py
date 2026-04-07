from __future__ import annotations

from src.config import ClassificationConfig
from src.pipeline.models import TextBlock
from src.utils.geometry import horizontal_overlap_ratio


def _pair_score(headline: TextBlock, content: TextBlock, page_h: int) -> float:
    v_gap = max(0, content.bbox_px[1] - headline.bbox_px[3])
    if v_gap > page_h * 0.5:
        return -1e9
    h_overlap = horizontal_overlap_ratio(headline.bbox_px, content.bbox_px)
    center_dist = abs(
        ((headline.bbox_px[0] + headline.bbox_px[2]) / 2.0)
        - ((content.bbox_px[0] + content.bbox_px[2]) / 2.0)
    )
    return h_overlap * 2.0 - (center_dist / max(1.0, page_h)) - (v_gap / max(1.0, page_h))


def assign_article_groups(blocks: list[TextBlock], page_h: int, cfg: ClassificationConfig) -> None:
    if not blocks:
        return

    headlines = [b for b in blocks if b.role == "headline"]
    contents = [b for b in blocks if b.role == "content"]

    used_content_ids: set[int] = set()
    group_idx = 1

    for headline in sorted(headlines, key=lambda b: (b.bbox_px[1], b.bbox_px[0])):
        candidates = []
        for c in contents:
            cid = id(c)
            if cid in used_content_ids:
                continue
            if c.bbox_px[1] < headline.bbox_px[1]:
                continue
            if (c.bbox_px[1] - headline.bbox_px[3]) > page_h * cfg.pair_max_vertical_ratio:
                continue
            candidates.append(c)

        if not candidates:
            continue

        best = max(candidates, key=lambda c: _pair_score(headline, c, page_h))
        if _pair_score(headline, best, page_h) < -0.15:
            continue

        gid = f"a{group_idx}"
        group_idx += 1
        headline.article_group_id = gid
        best.article_group_id = gid
        used_content_ids.add(id(best))

    for block in sorted(blocks, key=lambda b: (b.bbox_px[1], b.bbox_px[0])):
        if block.article_group_id:
            continue
        block.article_group_id = f"a{group_idx}"
        group_idx += 1
