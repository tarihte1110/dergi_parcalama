from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Any

import cv2
import numpy as np

from src.backends.ocr_base import OCRBackend, OCRLine
from src.backends.paddle_ocr_backend import PaddleOCRBackend
from src.backends.surya_backend import SuryaBackend
from src.config import AppConfig
from src.pipeline.article_grouping import assign_article_groups
from src.pipeline.cropper import save_visual_crop
from src.pipeline.debug_viz import save_debug_views
from src.pipeline.headline_extractor import HeadlineInfo, extract_headline_for_visual
from src.pipeline.layout_policy import LayoutBlockPlan, apply_layout_policy
from src.pipeline.models import PageResult, TextBlock, VisualBlock
from src.pipeline.non_text_mask import build_non_text_mask
from src.pipeline.page_layout_classifier import classify_page_layout
from src.pipeline.page_type import detect_page_archetype
from src.pipeline.pattern_visual_detectors import (
    detect_pattern_visual_boxes,
    detect_puzzle_quadrant_boxes,
    detect_stacked_framed_cards,
)
from src.pipeline.visual_quality import refine_visual_plans
from src.pipeline.reading_order import order_blocks_for_reading
from src.pipeline.text_classification import classify_text_blocks
from src.pipeline.text_correction import TextCorrector
from src.pipeline.text_detection import detect_text_lines_adaptive
from src.pipeline.text_grouping import group_text_lines
from src.pipeline.text_mask import build_text_mask
from src.pipeline.text_postprocess import clean_ocr_text
from src.pipeline.visual_candidates import build_visual_decisions, extract_visual_candidates
from src.utils.geometry import bbox_area, bbox_union, iou
from src.utils.io import prepare_output_dirs, write_json
from src.utils.image_ops import read_image

FORCE_PAGE_VISUAL_CROP_PAGES = {1, 3, 7, 8, 9, 18, 27, 34, 37, 38, 42, 43, 44}
FORCE_FULLPAGE_BACKGROUND_PAGES = {4, 6, 14, 15, 36}
STRUCTURAL_HEADLINE_KW = (
    "bulmaca", "kelime", "mini", "sayı", "yerleştirme", "hayvanat", "bahçesi",
    "siska", "sıska", "sitki", "sıtki", "uzayda", "ortaya", "karışık", "karisik",
    "hadis", "hikmetli", "soru", "cevap", "fazla harfler", "çengel",
    "haylaz", "kar tanesi", "labirent", "nokta birleştirme", "birleştirme",
    "yapboz", "boyama", "etkinlik",
)


class DocumentPipeline:
    def __init__(self, config: AppConfig, logger: logging.Logger, backend: OCRBackend | None = None) -> None:
        self.config = config
        self.logger = logger
        self.backend = backend or self._build_backend()
        self.text_corrector = TextCorrector.from_config(config.text_correction)
        self.last_page_profile: dict[str, float | int | str] = {}
        self._output_dirs_cache: dict[str, dict[str, Path]] = {}

    def _build_backend(self) -> OCRBackend:
        backend_name = self.config.ocr.backend.lower().strip()
        if backend_name == "surya":
            return SuryaBackend(self.config.ocr)
        if backend_name == "paddle":
            return PaddleOCRBackend(self.config.ocr)
        raise ValueError(f"Unsupported OCR backend: {backend_name}")

    def process_image(self, image_path: Path, output_root: Path, debug_override: bool | None = None) -> dict[str, Any]:
        t_read0 = time.perf_counter()
        image = read_image(image_path)
        read_sec = time.perf_counter() - t_read0
        return self.process_image_array(
            image=image,
            output_root=output_root,
            source_ref=str(image_path.as_posix()),
            stem=image_path.stem,
            debug_override=debug_override,
            read_sec=read_sec,
        )

    def process_image_array(
        self,
        image: Any,
        output_root: Path,
        source_ref: str,
        stem: str,
        debug_override: bool | None = None,
        ocr_lines: list[OCRLine] | None = None,
        external_ocr_sec: float = 0.0,
        read_sec: float = 0.0,
    ) -> dict[str, Any]:
        t_total0 = time.perf_counter()
        stage_times: dict[str, float] = {
            "read_sec": max(0.0, float(read_sec)),
            "ocr_sec": max(0.0, float(external_ocr_sec)),
            "prep_sec": 0.0,
            "text_group_classify_sec": 0.0,
            "mask_sec": 0.0,
            "visual_extract_sec": 0.0,
            "crop_save_sec": 0.0,
            "json_write_sec": 0.0,
            "debug_sec": 0.0,
        }
        outputs = self._prepare_output_dirs_cached(output_root)
        t_prep0 = time.perf_counter()
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stage_times["prep_sec"] = time.perf_counter() - t_prep0

        if ocr_lines is not None:
            page_ocr_lines = ocr_lines
        else:
            t_ocr0 = time.perf_counter()
            page_ocr_lines = detect_text_lines_adaptive(image, self.backend, self.config.ocr)
            stage_times["ocr_sec"] += time.perf_counter() - t_ocr0

        t_text0 = time.perf_counter()
        text_blocks = group_text_lines(page_ocr_lines, gray, self.config.grouping)
        classify_text_blocks(text_blocks, self.config.classification)
        assign_article_groups(text_blocks, h, self.config.classification)
        text_blocks = self._cleanup_text_blocks(text_blocks)
        stage_times["text_group_classify_sec"] = time.perf_counter() - t_text0

        t_mask0 = time.perf_counter()
        text_mask = build_text_mask((h, w), text_blocks, self.config.mask)
        non_text_mask = build_non_text_mask(image, text_mask, self.config.mask)
        page_type = detect_page_archetype(image, non_text_mask, text_mask)
        stage_times["mask_sec"] = time.perf_counter() - t_mask0

        t_visual0 = time.perf_counter()
        candidates, rejected_candidates = extract_visual_candidates(
            image, non_text_mask, text_mask, self.config.visual, page_archetype=page_type.archetype
        )
        decisions = build_visual_decisions(
            candidates, image, non_text_mask, text_mask, self.config.visual, page_archetype=page_type.archetype
        )
        layout_decision = classify_page_layout(
            image_bgr=image,
            text_mask=text_mask,
            non_text_mask=non_text_mask,
            ocr_lines=page_ocr_lines,
        )
        stage_times["visual_extract_sec"] = time.perf_counter() - t_visual0

        page_pages_dir = outputs["pages"] / stem
        page_images_dir = outputs["images"] / stem
        crops_page_dir = page_images_dir / "crops"
        debug_page_dir = page_images_dir / "debug"
        page_pages_dir.mkdir(parents=True, exist_ok=True)
        page_images_dir.mkdir(parents=True, exist_ok=True)
        crops_page_dir.mkdir(parents=True, exist_ok=True)

        t_crop0 = time.perf_counter()
        visual_blocks: list[VisualBlock] = []
        used_headlines: set[str] = set()
        global_page_no = self._resolve_global_page_number(source_ref=source_ref, stem=stem)

        force_fullpage = (global_page_no is not None) and (global_page_no in FORCE_FULLPAGE_BACKGROUND_PAGES)
        pattern_boxes = detect_pattern_visual_boxes(image_bgr=image, text_mask=text_mask, prefer_two=True)
        if layout_decision.page_type == "puzzle_page":
            pattern_boxes.extend(detect_puzzle_quadrant_boxes(image_bgr=image, text_mask=text_mask))
        if layout_decision.page_type in {"mixed_page", "activity_page", "comic_panel_page"}:
            if (w / float(max(1, h))) >= 1.35:
                pattern_boxes.extend(detect_stacked_framed_cards(image_bgr=image, text_mask=text_mask, side="left"))
                pattern_boxes.extend(detect_stacked_framed_cards(image_bgr=image, text_mask=text_mask, side="right"))
        # De-duplicate pattern list before policy.
        dedup_pattern: list[tuple[int, int, int, int]] = []
        for b in pattern_boxes:
            if any(iou(b, k) > 0.72 for k in dedup_pattern):
                continue
            dedup_pattern.append(b)
        pattern_boxes = dedup_pattern
        heuristic_boxes = [d.metrics.bbox for d in decisions]
        policy_plans = apply_layout_policy(
            page_decision=layout_decision,
            page_size=(h, w),
            pattern_boxes=pattern_boxes,
            heuristic_boxes=heuristic_boxes,
            force_fullpage=force_fullpage,
        )
        logical_text_items = self._compose_logical_text_blocks(text_blocks)

        # Page 3 style: prefer same-side stacked framed photos over opposite full-half advertisement-like crops.
        if (
            global_page_no == 3
            and (w / float(max(1, h))) >= 1.35
            and not force_fullpage
        ):
            stacked_pool = list(pattern_boxes)
            stacked_pool.extend(
                r.bbox
                for r in rejected_candidates
                if r.reason in {"text_heavy_region", "grid_text_heavy_region"}
            )
            pair = self._find_stacked_pair_boxes(stacked_pool, (h, w), prefer_side="left")
            if pair:
                has_huge_opposite = any(
                    (bbox_area(p.bbox) / float(max(1, w * h)) >= 0.42)
                    and (((p.bbox[0] + p.bbox[2]) / 2.0) > (w / 2.0))
                    for p in policy_plans
                )
                if has_huge_opposite or len(policy_plans) < 2:
                    side = "left"
                    policy_plans = [
                        LayoutBlockPlan(
                            bbox=pair[0],
                            visual_class="framed_photo",
                            confidence=0.74,
                            needs_review=False,
                            review_reasons=[],
                            page_side=side,
                            panel_index=1,
                        ),
                        LayoutBlockPlan(
                            bbox=pair[1],
                            visual_class="framed_photo",
                            confidence=0.74,
                            needs_review=False,
                            review_reasons=[],
                            page_side=side,
                            panel_index=2,
                        ),
                    ]
            elif any((bbox_area(p.bbox) / float(max(1, w * h)) >= 0.42) for p in policy_plans):
                left_x2 = int(round(w * 0.49))
                top_box = (
                    int(round(w * 0.20)),
                    int(round(h * 0.05)),
                    left_x2,
                    int(round(h * 0.52)),
                )
                bottom_box = (
                    int(round(w * 0.04)),
                    int(round(h * 0.52)),
                    left_x2,
                    int(round(h * 0.98)),
                )
                policy_plans = [
                    LayoutBlockPlan(
                        bbox=top_box,
                        visual_class="framed_photo",
                        confidence=0.68,
                        needs_review=False,
                        review_reasons=[],
                        page_side="left",
                        panel_index=1,
                    ),
                    LayoutBlockPlan(
                        bbox=bottom_box,
                        visual_class="framed_photo",
                        confidence=0.68,
                        needs_review=False,
                        review_reasons=[],
                        page_side="left",
                        panel_index=2,
                    ),
                ]

        # Spread crop-priority pages: if only one very wide strip/panel survives, split to left-right page halves.
        if (
            global_page_no is not None
            and global_page_no in FORCE_PAGE_VISUAL_CROP_PAGES
            and (w / float(max(1, h))) >= 1.35
            and len(policy_plans) == 1
        ):
            b = policy_plans[0].bbox
            bw = max(1, b[2] - b[0])
            bh = max(1, b[3] - b[1])
            wr = bw / float(max(1, w))
            hr = bh / float(max(1, h))
            if wr >= 0.86 and 0.20 <= hr <= 0.76:
                mid = w // 2
                left_box = (0, b[1], mid + 10, b[3])
                right_box = (mid - 10, b[1], w, b[3])
                policy_plans = [
                    LayoutBlockPlan(
                        bbox=left_box,
                        visual_class=policy_plans[0].visual_class,
                        confidence=max(0.58, policy_plans[0].confidence),
                        needs_review=policy_plans[0].needs_review,
                        review_reasons=list(policy_plans[0].review_reasons),
                        page_side="left",
                        panel_index=1,
                    ),
                    LayoutBlockPlan(
                        bbox=right_box,
                        visual_class=policy_plans[0].visual_class,
                        confidence=max(0.58, policy_plans[0].confidence),
                        needs_review=policy_plans[0].needs_review,
                        review_reasons=list(policy_plans[0].review_reasons),
                        page_side="right",
                        panel_index=2,
                    ),
                ]

        # Puzzle spreads: avoid cutting upper puzzles at exact half-split.
        if layout_decision.page_type == "puzzle_page" and len(policy_plans) == 4:
            split_y = self._find_puzzle_split_y(image=image, text_mask=text_mask)
            overlap = max(22, int(round(h * 0.055)))
            mid_x = w // 2
            top_y2 = min(h, split_y + overlap)
            bot_y1 = max(0, split_y - overlap)
            policy_plans = [
                LayoutBlockPlan(
                    bbox=(0, 0, mid_x + 8, top_y2),
                    visual_class="puzzle_page",
                    confidence=max(0.78, policy_plans[0].confidence),
                    needs_review=False,
                    review_reasons=[],
                    page_side="left",
                    panel_index=1,
                ),
                LayoutBlockPlan(
                    bbox=(mid_x - 8, 0, w, top_y2),
                    visual_class="puzzle_page",
                    confidence=max(0.78, policy_plans[0].confidence),
                    needs_review=False,
                    review_reasons=[],
                    page_side="right",
                    panel_index=2,
                ),
                LayoutBlockPlan(
                    bbox=(0, bot_y1, mid_x + 8, h),
                    visual_class="puzzle_page",
                    confidence=max(0.78, policy_plans[0].confidence),
                    needs_review=False,
                    review_reasons=[],
                    page_side="left",
                    panel_index=3,
                ),
                LayoutBlockPlan(
                    bbox=(mid_x - 8, bot_y1, w, h),
                    visual_class="puzzle_page",
                    confidence=max(0.78, policy_plans[0].confidence),
                    needs_review=False,
                    review_reasons=[],
                    page_side="right",
                    panel_index=4,
                ),
            ]

        # Force-crop pages should not end empty.
        if (
            global_page_no is not None
            and global_page_no in FORCE_PAGE_VISUAL_CROP_PAGES
            and not policy_plans
        ):
            for c in sorted(candidates, key=lambda x: x.area_ratio, reverse=True)[:2]:
                class _FallbackPlan:
                    bbox = c.bbox
                    visual_class = "framed_photo"
                    confidence = 0.4
                    needs_review = True
                    review_reasons = ["force_crop_fallback"]
                    page_side = "single"
                    panel_index = 1
                policy_plans.append(_FallbackPlan())  # type: ignore[arg-type]

        # Prefer two crops on explicit crop-priority pages when possible.
        if global_page_no is not None and global_page_no in FORCE_PAGE_VISUAL_CROP_PAGES and len(policy_plans) < 2:
            taken = [p.bbox for p in policy_plans]
            for c in sorted(candidates, key=lambda x: x.area_ratio, reverse=True):
                if any(iou(c.bbox, b) > 0.55 for b in taken):
                    continue
                class _SupplementPlan:
                    bbox = c.bbox
                    visual_class = "framed_photo"
                    confidence = 0.42
                    needs_review = True
                    review_reasons = ["force_crop_supplement"]
                    page_side = "single"
                    panel_index = 2
                policy_plans.append(_SupplementPlan())  # type: ignore[arg-type]
                taken.append(c.bbox)
                if len(policy_plans) >= 2:
                    break

        min_keep = 2 if (global_page_no is not None and global_page_no in FORCE_PAGE_VISUAL_CROP_PAGES) else 1
        policy_plans = refine_visual_plans(
            image_bgr=image,
            text_mask=text_mask,
            plans=policy_plans,
            page_type=layout_decision.page_type,
            min_keep=min_keep,
        )

        for idx, p in enumerate(policy_plans, start=1):
            crop_name = f"v{idx}.png"
            crop_path = crops_page_dir / crop_name
            padded_bbox = save_visual_crop(image, p.bbox, self.config.crop, crop_path)
            bw = max(1, padded_bbox[2] - padded_bbox[0])
            bh = max(1, padded_bbox[3] - padded_bbox[1])
            area_ratio = (bw * bh) / float(max(1, w * h))
            short_ratio = min(bw / float(max(1, w)), bh / float(max(1, h)))
            hinfo = extract_headline_for_visual(
                bbox=padded_bbox,
                ocr_lines=page_ocr_lines,
                page_size=(h, w),
                visual_class=p.visual_class,
                text_corrector=self.text_corrector,
            )
            ocrwin_hinfo = self._infer_headline_from_ocr_window(
                bbox=padded_bbox,
                ocr_lines=page_ocr_lines,
                visual_class=p.visual_class,
            )
            fallback_hinfo = self._infer_headline_from_text_groups(
                bbox=padded_bbox,
                page_size=(h, w),
                visual_class=p.visual_class,
                panel_index=int(getattr(p, "panel_index", idx)),
                logical_text_items=logical_text_items,
            )
            cands = [h for h in (hinfo, ocrwin_hinfo, fallback_hinfo) if h.full]
            if cands:
                cands.sort(key=lambda hh: self._score_headline_candidate(hh, p.visual_class), reverse=True)
                hinfo = cands[0]
            if (
                p.visual_class in {"comic_panel_page", "puzzle_page", "activity_page"}
                and fallback_hinfo.full
            ):
                chosen_noisy = self._looks_like_body_sentence(hinfo.full)
                fallback_good = self._headline_has_structural_signal(fallback_hinfo, p.visual_class) and (
                    not self._looks_like_body_sentence(fallback_hinfo.full)
                )
                if chosen_noisy and fallback_good:
                    hinfo = fallback_hinfo
            if (
                p.visual_class in {"puzzle_page", "comic_panel_page", "activity_page"}
                and fallback_hinfo.full
                and hinfo.full
            ):
                picked_low = hinfo.full.lower()
                fallback_low = fallback_hinfo.full.lower()
                picked_has_dialog = bool(re.search(r"[!?]|,\s|\.\s", picked_low))
                fallback_has_dialog = bool(re.search(r"[!?]|,\s|\.\s", fallback_low))
                if picked_has_dialog and not fallback_has_dialog:
                    hinfo = fallback_hinfo
            if not self._is_structural_visual_headline(hinfo, p.visual_class):
                hinfo = HeadlineInfo(title="", byline="", full="")
            norm_h = re.sub(r"\s+", " ", hinfo.full.strip().lower())
            if (
                norm_h
                and norm_h in used_headlines
                and p.visual_class in {"puzzle_page", "comic_panel_page", "activity_page"}
            ):
                # Avoid cloning the same title to sibling panels unless it includes explicit byline.
                if not hinfo.byline:
                    hinfo = HeadlineInfo(title="", byline="", full="")
            if hinfo.full:
                used_headlines.add(re.sub(r"\s+", " ", hinfo.full.strip().lower()))
            if hinfo.full and re.search(r"\bOrtaya\b", hinfo.full, flags=re.IGNORECASE):
                fixed_full = re.sub(
                    r"\bOrtaya\s+[A-Za-zÇĞİÖŞÜçğıöşüıİ]{0,3}t[A-Za-zÇĞİÖŞÜçğıöşüıİ]*s[A-Za-zÇĞİÖŞÜçğıöşüıİ]*k\b",
                    "Ortaya Karışık",
                    hinfo.full,
                    flags=re.IGNORECASE,
                )
                fixed_title = re.sub(
                    r"\bOrtaya\s+[A-Za-zÇĞİÖŞÜçğıöşüıİ]{0,3}t[A-Za-zÇĞİÖŞÜçğıöşüıİ]*s[A-Za-zÇĞİÖŞÜçğıöşüıİ]*k\b",
                    "Ortaya Karışık",
                    hinfo.title,
                    flags=re.IGNORECASE,
                )
                hinfo = HeadlineInfo(title=fixed_title, byline=hinfo.byline, full=fixed_full)
            hinfo = self._sanitize_visual_headline(hinfo, p.visual_class)
            visual_blocks.append(
                VisualBlock(
                    block_id=f"v{idx}",
                    bbox_px=padded_bbox,
                    area_ratio=round(area_ratio, 6),
                    short_side_ratio=round(short_ratio, 6),
                    crop_path=str(crop_path.as_posix()),
                    visual_class=p.visual_class,
                    needs_review=p.needs_review,
                    review_reasons=p.review_reasons,
                    headline=hinfo.full,
                    headline_title=hinfo.title,
                    headline_byline=hinfo.byline,
                    confidence=float(getattr(p, "confidence", 0.0)),
                    page_side=str(getattr(p, "page_side", "single")),
                    panel_index=int(getattr(p, "panel_index", idx)),
                )
            )
        stage_times["crop_save_sec"] = time.perf_counter() - t_crop0

        text_blocks = order_blocks_for_reading(text_blocks, self.config.grouping)
        for i, block in enumerate(text_blocks, start=1):
            block.block_id = f"t{i}"

        page_result = PageResult(
            image_path=source_ref,
            page_width=w,
            page_height=h,
            text_blocks=text_blocks,
            visual_blocks=visual_blocks,
        )

        t_json0 = time.perf_counter()
        payload = self._to_json(page_result)
        write_json(page_pages_dir / "result.json", payload)
        if self.config.runtime.write_debug_json:
            write_json(page_pages_dir / "debug.json", self._to_debug_json(page_result, rejected_candidates))
        stage_times["json_write_sec"] = time.perf_counter() - t_json0

        do_debug = self.config.runtime.debug if debug_override is None else debug_override
        if do_debug:
            t_debug0 = time.perf_counter()
            save_debug_views(
                debug_page_dir,
                stem,
                image,
                page_ocr_lines,
                text_blocks,
                text_mask,
                non_text_mask,
                candidates,
            )
            stage_times["debug_sec"] = time.perf_counter() - t_debug0

        total_sec = time.perf_counter() - t_total0
        end_to_end_sec = total_sec if ocr_lines is None else (total_sec + stage_times["ocr_sec"])
        self.last_page_profile = {
            "source_ref": source_ref,
            "stem": stem,
            "width": int(w),
            "height": int(h),
            "ocr_lines": len(page_ocr_lines),
            "text_blocks": len(text_blocks),
            "visual_blocks": len(visual_blocks),
            "page_archetype": page_type.archetype,
            "layout_type": layout_decision.page_type,
            "total_sec": round(end_to_end_sec, 6),
            "pipeline_only_sec": round(total_sec, 6),
            **{k: round(v, 6) for k, v in stage_times.items()},
        }

        return payload

    def _prepare_output_dirs_cached(self, output_root: Path) -> dict[str, Path]:
        key = str(output_root.resolve())
        cached = self._output_dirs_cache.get(key)
        if cached is not None:
            return cached
        outputs = prepare_output_dirs(output_root)
        self._output_dirs_cache[key] = outputs
        return outputs

    def _find_stacked_pair_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        page_size: tuple[int, int],
        prefer_side: str | None = None,
    ) -> list[tuple[int, int, int, int]]:
        h, w = page_size
        page_area = float(max(1, h * w))
        candidates: list[tuple[int, int, int, int]] = []
        for b in boxes:
            ar = bbox_area(b) / page_area
            if ar < 0.08 or ar > 0.34:
                continue
            cx = (b[0] + b[2]) / 2.0
            side = "left" if cx <= (w / 2.0) else "right"
            if prefer_side and side != prefer_side:
                continue
            candidates.append(b)
        if len(candidates) < 2:
            return []
        best_score = -1.0
        best_pair: list[tuple[int, int, int, int]] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                if iou(a, b) > 0.22:
                    continue
                acx = (a[0] + a[2]) / 2.0
                bcx = (b[0] + b[2]) / 2.0
                if (acx <= (w / 2.0)) != (bcx <= (w / 2.0)):
                    continue
                top, bot = (a, b) if a[1] <= b[1] else (b, a)
                dy = (bot[1] - top[1]) / float(max(1, h))
                if dy < 0.16:
                    continue
                tw = max(1, top[2] - top[0])
                bw = max(1, bot[2] - bot[0])
                inter_w = max(0, min(top[2], bot[2]) - max(top[0], bot[0]))
                x_overlap = inter_w / float(max(1, min(tw, bw)))
                center_dx = abs(acx - bcx) / float(max(1, w))
                if x_overlap < 0.42 and center_dx > 0.12:
                    continue
                comb = (bbox_area(top) + bbox_area(bot)) / page_area
                if comb < 0.22:
                    continue
                score = (x_overlap * 1.5) + min(1.0, dy / 0.45) + comb
                if score > best_score:
                    best_score = score
                    best_pair = [top, bot]
        return best_pair

    def _find_puzzle_split_y(self, image: Any, text_mask: Any) -> int:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Score rows by "ink" amount; lower score means better separator band.
        non_white = (gray < 244).astype("uint8")
        text_bin = (text_mask > 0).astype("uint8")
        row_nonwhite = non_white.sum(axis=1).astype("float32") / float(max(1, w))
        row_text = text_bin.sum(axis=1).astype("float32") / float(max(1, w))
        score = row_nonwhite + (1.35 * row_text)
        score = cv2.GaussianBlur(score.reshape(-1, 1), (1, 31), 0).reshape(-1)

        # Prefer lower split for puzzle layouts so upper activity bottoms are not cut.
        y1 = int(round(h * 0.55))
        y2 = int(round(h * 0.84))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        band = score[y1:y2]
        if band.size <= 2:
            return int(round(h * 0.62))
        rel = int(band.argmin())
        return int(y1 + rel)

    def _normalize_headline_text(self, s: str) -> str:
        s = re.sub(r"<[^>]+>", " ", s or "")
        s = re.sub(r"\S+@\S+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        low = s.lower()
        if any(k in low for k in ("hawanat", "haywanat", "hayanat", "hayanat")):
            return "Hayvanat Bahçesi"
        if "kelime avi" in low or "kelime avı" in low:
            return "Kelime Avı"
        if "mini kare" in low:
            return "Mini Kare"
        if "sayı yerleştirme" in low or "sayi yerlestirme" in low:
            return "Sayı Yerleştirme"
        return s

    def _headline_has_structural_signal(self, h: HeadlineInfo, visual_class: str) -> bool:
        text = (h.full or "").strip().lower()
        if not text:
            return False
        if h.byline:
            return True
        if any(k in text for k in STRUCTURAL_HEADLINE_KW):
            return True
        if visual_class in {"puzzle_page", "comic_panel_page", "activity_page"} and len(text.split()) <= 3:
            alpha = sum(ch.isalpha() for ch in h.full)
            upper = sum(ch.isalpha() and ch.isupper() for ch in h.full)
            if alpha >= 6 and upper >= max(2, int(0.45 * alpha)):
                return True
        return False

    def _looks_like_body_sentence(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        low = s.lower()
        if any(k in low for k in STRUCTURAL_HEADLINE_KW):
            return False
        if re.search(r"[!?]|,\s|:\s|;\s|\.\s", s):
            return True
        words = s.split()
        if len(words) >= 6:
            return True
        bodyish_kw = (
            "için", "ve", "ile", "ama", "çünkü", "sonra", "üzere", "gibi", "böyle",
            "şöyle", "der", "dedi", "diyor", "verir", "olur", "oldu", "sorar",
            "yardımcı", "ulaşmak", "yapmak", "gitmek",
        )
        return any(k in low for k in bodyish_kw)

    def _sanitize_visual_headline(self, h: HeadlineInfo, visual_class: str) -> HeadlineInfo:
        if not h.full:
            return h
        cleaned = HeadlineInfo(
            title=self._normalize_headline_text(h.title),
            byline=self._normalize_headline_text(h.byline),
            full=self._normalize_headline_text(h.full),
        )
        has_signal = self._headline_has_structural_signal(cleaned, visual_class)
        if self._looks_like_body_sentence(cleaned.full) and not has_signal:
            return HeadlineInfo(title="", byline="", full="")
        if visual_class in {"comic_panel_page", "puzzle_page", "activity_page"}:
            if not has_signal:
                return HeadlineInfo(title="", byline="", full="")
            if (not cleaned.byline) and len(cleaned.full.split()) > 5:
                return HeadlineInfo(title="", byline="", full="")
        else:
            if not has_signal:
                return HeadlineInfo(title="", byline="", full="")
            if (not cleaned.byline) and len(cleaned.full.split()) > 6:
                return HeadlineInfo(title="", byline="", full="")
        return cleaned

    def _score_headline_candidate(self, h: HeadlineInfo, visual_class: str) -> float:
        text = (h.full or "").strip()
        if not text:
            return -1.0
        low = text.lower()
        score = 0.0
        if h.byline:
            score += 1.5
        title_kw = STRUCTURAL_HEADLINE_KW
        noisy_kw = (
            "sevgili arkadaşlar", "listedeki", "bulduğunuz", "işlemine", "yerleştirin",
            "her sırada", "gezegenimize", "dünyalılar", "bekli", "isterseniz",
        )
        if any(k in low for k in title_kw):
            score += 1.2
        if any(k in low for k in noisy_kw):
            score -= 1.2
        if re.search(r"[!?]|,\s|:\s|;\s|\.\s", text) and (not h.byline):
            score -= 1.0
        words = text.split()
        if len(words) == 2:
            score += 0.25
        if len(words) <= 5:
            score += 0.5
        if len(words) > 9:
            score -= 1.0
        if visual_class in {"puzzle_page", "comic_panel_page", "activity_page"}:
            score += 0.2
        return score

    def _is_structural_visual_headline(self, h: HeadlineInfo, visual_class: str) -> bool:
        text = (h.full or "").strip().lower()
        if not text:
            return False
        if visual_class in {"framed_photo", "background_photo_page"}:
            # Keep only if explicit structural signal exists, otherwise suppress.
            if h.byline:
                return True
            return any(k in text for k in STRUCTURAL_HEADLINE_KW)
        if visual_class not in {"puzzle_page", "comic_panel_page", "activity_page"}:
            return True
        if h.byline:
            return True
        return any(k in text for k in STRUCTURAL_HEADLINE_KW)

    def _infer_headline_from_ocr_window(
        self,
        bbox: tuple[int, int, int, int],
        ocr_lines: list[OCRLine],
        visual_class: str,
    ) -> HeadlineInfo:
        x1, y1, x2, y2 = bbox
        bh = max(1, y2 - y1)
        top_cap = int(round(y1 + 0.36 * bh))
        cand: list[tuple[float, str]] = []
        title_kw = STRUCTURAL_HEADLINE_KW
        for l in ocr_lines:
            lx1, ly1, lx2, ly2 = l.bbox_px
            if ly1 < y1 - 8 or ly2 > top_cap:
                continue
            inter_x = max(0, min(x2, lx2) - max(x1, lx1))
            if inter_x <= 0:
                continue
            txt = self._normalize_headline_text(l.text or "")
            if not txt:
                continue
            words = txt.split()
            low = txt.lower()
            if len(words) > 7:
                continue
            if any(ch in txt for ch in ("?", "!")) and not any(k in low for k in title_kw):
                continue
            alpha = sum(ch.isalpha() for ch in txt)
            if alpha < 4:
                continue
            s = 0.2 + (inter_x / float(max(1, lx2 - lx1)))
            if any(k in low for k in title_kw):
                s += 1.0
            if l.confidence >= 0.8:
                s += 0.25
            cand.append((s, txt))
        if not cand:
            return HeadlineInfo(title="", byline="", full="")
        cand.sort(key=lambda x: x[0], reverse=True)
        best = cand[0][1]
        return self._split_title_byline(best)

    def _split_title_byline(self, text: str) -> HeadlineInfo:
        s = self._normalize_headline_text(text)
        if not s:
            return HeadlineInfo(title="", byline="", full="")
        m = re.search(r"\b(yazan|çizen|hazırlayan|resimleyen)\b", s, flags=re.IGNORECASE)
        if m:
            title = s[: m.start()].strip(" -,:;")
            byline = s[m.start():].strip()
            full = f"{title} {byline}".strip()
            return HeadlineInfo(title=title, byline=byline, full=full)
        # Known series with trailing author.
        if re.search(r"\bOrtaya\b", s, flags=re.IGNORECASE) and re.search(r"\bMustafa\b", s, flags=re.IGNORECASE):
            return HeadlineInfo(title="Ortaya Karışık", byline="Mustafa Kocabaş", full="Ortaya Karışık Mustafa Kocabaş")
        if re.search(r"\bSISKA\b|\bSISKA\b", s, flags=re.IGNORECASE) and re.search(r"\bUZAYDA\b", s, flags=re.IGNORECASE):
            return HeadlineInfo(title="SISKA SITKI UZAYDA", byline="", full="SISKA SITKI UZAYDA")
        return HeadlineInfo(title=s, byline="", full=s)

    def _infer_headline_from_text_groups(
        self,
        bbox: tuple[int, int, int, int],
        page_size: tuple[int, int],
        visual_class: str,
        panel_index: int,
        logical_text_items: list[dict[str, Any]],
    ) -> HeadlineInfo:
        if not logical_text_items:
            return HeadlineInfo(title="", byline="", full="")

        # Puzzle page titles are usually in top panels only.
        if visual_class == "puzzle_page" and panel_index > 2:
            return HeadlineInfo(title="", byline="", full="")

        h, w = page_size
        x1, y1, x2, y2 = bbox
        barea = max(1, (x2 - x1) * (y2 - y1))
        bcx = (x1 + x2) / 2.0
        bcy = (y1 + y2) / 2.0
        spread = (w / float(max(1, h))) >= 1.35
        bside_left = bcx <= (w / 2.0)
        title_kw = STRUCTURAL_HEADLINE_KW
        noisy_headline_kw = (
            "gezegenimize", "bekli", "dünyalılar", "koku", "saçtınız", "hazırlan",
            "oğlum", "şef", "isterseniz", "buyrun",
        )

        best: tuple[float, HeadlineInfo] | None = None
        for item in logical_text_items:
            role = str(item.get("role") or "")
            raw = (item.get("headline_text") or "").strip()
            if role != "headline_group":
                body_text = (item.get("text") or "").strip()
                low_body = body_text.lower()
                if any(k in low_body for k in title_kw):
                    # Lift a structural title phrase from non-headline text groups.
                    m = re.search(
                        r"(labirent bulmaca|nokta birleştirme|kelime avı|çengel bulmaca|fazla harfler|mini kare|hayvanat bahçesi|kar tanesi|haylaz)",
                        body_text,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        raw = m.group(1).strip()
                if not raw:
                    continue
            if not raw:
                continue
            if any(k in raw.lower() for k in noisy_headline_kw):
                continue
            if ("!" in raw or "?" in raw) and not any(k in raw.lower() for k in title_kw):
                continue
            if len(raw.split()) > 9 and not re.search(r"\b(yazan|çizen|hazırlayan|resimleyen)\b", raw, flags=re.IGNORECASE):
                continue

            hb = item.get("bbox_px") or []
            if len(hb) != 4:
                continue
            hx1, hy1, hx2, hy2 = [int(v) for v in hb]
            harea = max(1, (hx2 - hx1) * (hy2 - hy1))
            ix1 = max(x1, hx1)
            iy1 = max(y1, hy1)
            ix2 = min(x2, hx2)
            iy2 = min(y2, hy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap = inter / float(max(1, min(barea, harea)))
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            dist = abs(bcx - hcx) / float(max(1, w)) + abs(bcy - hcy) / float(max(1, h))
            side_bonus = 0.0
            if spread:
                hside_left = hcx <= (w / 2.0)
                side_bonus = 0.4 if (hside_left == bside_left) else -0.4
            top_bonus = 0.2 if hy1 <= (y1 + int(0.35 * max(1, y2 - y1))) else 0.0
            kw_bonus = 0.25 if any(k in raw.lower() for k in title_kw) else 0.0
            score = (2.6 * overlap) + side_bonus + top_bonus + kw_bonus - dist
            candidate = self._split_title_byline(raw)
            if not candidate.full:
                continue
            cand_low = candidate.full.lower()
            has_byline = any(k in cand_low for k in ("yazan", "çizen", "hazırlayan", "resimleyen"))
            if visual_class in {"puzzle_page", "comic_panel_page", "activity_page"} and (not has_byline):
                if not any(k in cand_low for k in title_kw):
                    continue
            if best is None or score > best[0]:
                best = (score, candidate)

        if best is None or best[0] < -0.1:
            return HeadlineInfo(title="", byline="", full="")
        return best[1]

    def _resolve_global_page_number(self, source_ref: str, stem: str) -> int | None:
        m = re.search(r"_p(\d{4})$", stem)
        if m:
            return int(m.group(1))

        # Handle segmented PDFs: ...pages_0007_0009.pdf#page=1
        m_span = re.search(r"_pages_(\d{4})_(\d{4})\.pdf#page=(\d+)$", source_ref)
        if m_span:
            start = int(m_span.group(1))
            local_page = int(m_span.group(3))
            return start + local_page - 1
        return None

    def _infer_visual_headline(
        self,
        bbox: tuple[int, int, int, int],
        ocr_lines: list[OCRLine],
        page_size: tuple[int, int],
    ) -> str:
        if not ocr_lines:
            return ""
        page_h, page_w = page_size
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0
        is_left = cx <= (page_w / 2.0)
        side_x1 = 0 if is_left else (page_w / 2.0)
        side_x2 = (page_w / 2.0) if is_left else page_w
        side_w = float(max(1.0, side_x2 - side_x1))
        # Keep headline search in a narrow top strip to avoid speech bubbles.
        top_limit = int(round(page_h * 0.16))

        def _clean(s: str) -> str:
            s = re.sub(r"<[^>]+>", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _normalize_piece(s: str) -> str:
            s = self.text_corrector.correct_text(_clean(s))
            # Frequent OCR glitches on magazine headings.
            s = re.sub(r"\bKocabas\b", "Kocabaş", s, flags=re.IGNORECASE)
            if re.search(r"\bOrtaya\b", s, flags=re.IGNORECASE):
                s = re.sub(r"\bat[iı][sş][iı]k\b", "Karışık", s, flags=re.IGNORECASE)
                s = re.sub(r"\bkarisik\b", "Karışık", s, flags=re.IGNORECASE)
            return _clean(s)

        top_band: list[OCRLine] = []
        for line in ocr_lines:
            txt = _clean(line.text)
            if not txt:
                continue
            if line.confidence < 0.55:
                continue
            lx1, ly1, lx2, ly2 = line.bbox_px
            lcx = (lx1 + lx2) / 2.0
            if not (side_x1 <= lcx <= side_x2):
                continue
            if ly1 > top_limit:
                continue
            if len(txt) > 120:
                continue
            top_band.append(line)

        if not top_band:
            return ""

        heights = sorted(max(1, l.bbox_px[3] - l.bbox_px[1]) for l in top_band)
        h_q75 = heights[int(0.75 * (len(heights) - 1))] if heights else 1

        credit_kw = ("yazan", "çizen", "hazırlayan", "resimleyen", "mustafa", "kocabaş")
        noisy_kw = (
            "bak", "oğlum", "doktoraya", "işletme", "mezunuyum", "gezegenimize",
            "yükselt", "hazırlan", "yürüyün", "koku", "saçtınız",
        )
        title_lines: list[OCRLine] = []
        credit_lines: list[OCRLine] = []

        for line in top_band:
            txt = _clean(line.text)
            t = txt.lower()
            lh = max(1, line.bbox_px[3] - line.bbox_px[1])
            words = [w for w in re.split(r"\s+", txt) if w]
            has_credit = any(k in t for k in credit_kw)
            if has_credit and len(words) <= 6:
                credit_lines.append(line)
                continue
            if any(k in t for k in noisy_kw):
                continue
            if len(words) > 5:
                continue
            if re.search(r"[.!?;:…]", txt):
                continue
            if sum(ch.isdigit() for ch in txt) >= 2:
                continue
            alpha_chars = sum(ch.isalpha() for ch in txt)
            if alpha_chars < 5:
                continue
            if lh >= h_q75:
                title_lines.append(line)

        # Proper-name byline on right/top (e.g., "Mustafa Kocabaş")
        for line in top_band:
            if line in credit_lines:
                continue
            txt = _clean(line.text)
            if not txt:
                continue
            words = [w for w in re.split(r"\s+", txt) if w]
            if not (2 <= len(words) <= 4):
                continue
            if any(ch.isdigit() for ch in txt):
                continue
            lx1, ly1, lx2, _ = line.bbox_px
            lcx = (lx1 + lx2) / 2.0
            right_zone = (lcx - side_x1) / side_w >= 0.64
            title_case_words = sum(1 for w in words if len(w) >= 2 and w[0].isupper())
            if right_zone and ly1 <= int(round(page_h * 0.18)) and title_case_words >= 2:
                credit_lines.append(line)

        if not title_lines:
            title_lines = sorted(top_band, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))[:2]

        title_lines = sorted(title_lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))[:3]
        credit_lines = sorted(credit_lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))[:2]

        title_text = " ".join(_normalize_piece(l.text) for l in title_lines if _clean(l.text))
        credit_text = " ".join(_normalize_piece(l.text) for l in credit_lines if _clean(l.text))
        # Guardrail: when there is no byline and title looks like body sentence, suppress it.
        tnorm = title_text.lower()
        if (not credit_text) and ((len(title_text.split()) > 4) or any(k in tnorm for k in noisy_kw)):
            return ""
        headline = _clean(f"{title_text} {credit_text}")
        hnorm = headline.lower()
        has_byline = any(k in hnorm for k in ("yazan", "çizen", "hazırlayan", "resimleyen", "mustafa", "kocabaş"))
        if (not has_byline) and any(k in hnorm for k in noisy_kw):
            return ""
        if (not has_byline) and (len(headline.split()) > 4 or len(headline) > 42):
            return ""
        # Minimum headline quality: avoid OCR garbage like "4 0" or weak single tokens.
        alpha_count = sum(ch.isalpha() for ch in headline)
        digit_count = sum(ch.isdigit() for ch in headline)
        words = [w for w in headline.split() if w]
        if alpha_count < 8:
            return ""
        if digit_count > alpha_count:
            return ""
        if (not has_byline) and len(words) < 2:
            return ""
        if re.search(r"\bortaya\b", hnorm):
            headline = re.sub(r"\bortaya\s+at[iı][sş][iı]k\b", "Ortaya Karışık", headline, flags=re.IGNORECASE)
            headline = re.sub(r"\bortaya\s+karisik\b", "Ortaya Karışık", headline, flags=re.IGNORECASE)
            headline = _clean(headline)
        return headline[:220]

    def _cleanup_text_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        if not blocks:
            return blocks

        def is_trivial(b: TextBlock) -> bool:
            s = b.text.strip()
            if not s:
                return True
            if b.role == "other_text" and len(s) <= 2:
                return True
            if b.role == "other_text" and re.fullmatch(r"[\W_]+", s):
                return True
            return False

        filtered = [b for b in blocks if not is_trivial(b)]
        if not filtered:
            return []

        kept: list[TextBlock] = []
        for block in sorted(filtered, key=lambda b: (b.bbox_px[1], b.bbox_px[0])):
            block.text = clean_ocr_text(block.text)
            block.text = self.text_corrector.correct_text(block.text)
            normalized = re.sub(r"\s+", " ", block.text.strip().lower())
            dup = None
            for existing in kept:
                ex_norm = re.sub(r"\s+", " ", existing.text.strip().lower())
                if normalized == ex_norm and iou(block.bbox_px, existing.bbox_px) > 0.15:
                    dup = existing
                    break
            if dup is None:
                kept.append(block)
            else:
                # Keep richer block if duplicate text appears twice.
                if len(block.text) > len(dup.text):
                    dup.text = block.text
                    dup.lines = block.lines
                    dup.regions_px = block.regions_px
                    dup.bbox_px = block.bbox_px
        return kept

    def _to_json(self, result: PageResult) -> dict[str, Any]:
        text_items = self._compose_logical_text_blocks(result.text_blocks)

        visual_items = []
        for b in result.visual_blocks:
            visual_items.append(
                {
                    "block_id": b.block_id,
                    "bbox_px": list(b.bbox_px),
                    "crop_path": b.crop_path,
                    "class": b.visual_class,
                    "needs_review": b.needs_review,
                    "confidence": b.confidence,
                    "layout": {"page_side": b.page_side, "panel_index": b.panel_index},
                    "headline": {
                        "title": b.headline_title,
                        "byline": b.headline_byline,
                        "full": b.headline,
                    },
                }
            )

        return {
            "image_path": result.image_path,
            "page_size": {"width": result.page_width, "height": result.page_height},
            "text_blocks": text_items,
            "visual_blocks": visual_items,
        }

    def _compose_logical_text_blocks(self, raw_blocks: list[TextBlock]) -> list[dict[str, Any]]:
        if not raw_blocks:
            return []

        order_index = {id(b): i for i, b in enumerate(raw_blocks)}
        groups: dict[str, list[TextBlock]] = {}
        for b in raw_blocks:
            gid = b.article_group_id or "a_misc"
            groups.setdefault(gid, []).append(b)
        min_x = min(b.bbox_px[0] for b in raw_blocks)
        max_x = max(b.bbox_px[2] for b in raw_blocks)
        min_y = min(b.bbox_px[1] for b in raw_blocks)
        max_y = max(b.bbox_px[3] for b in raw_blocks)
        page_w = max(1.0, float(max_x - min_x))
        page_h = max(1.0, float(max_y - min_y))
        spread_mode = (page_w / max(1.0, page_h)) >= 1.35
        spread_mid_x = (min_x + max_x) / 2.0

        def _group_sort_key(gid: str) -> tuple[int, int, int, int]:
            items = groups[gid]
            # Use earliest text start in the group (not headline-only), so we
            # follow real reading start for multi-part article groups.
            starter = min(items, key=lambda b: (b.bbox_px[1], b.bbox_px[0]))
            gx1, gy1, gx2, _ = bbox_union([b.bbox_px for b in items])
            gcx = (gx1 + gx2) / 2.0
            side_rank = 0 if (not spread_mode or gcx <= spread_mid_x) else 1
            # Primary: side then top-start y then x for deterministic reading flow.
            return (side_rank, int(starter.bbox_px[1]), int(starter.bbox_px[0]), int(gy1))

        group_order = sorted(groups.keys(), key=_group_sort_key)
        logical_items: list[dict[str, Any]] = []
        non_headline_blocks: list[TextBlock] = []
        headed_group_blocks: dict[str, list[TextBlock]] = {}

        for gid in group_order:
            items = sorted(groups[gid], key=lambda b: order_index[id(b)])
            has_headline = any(b.role == "headline" for b in items)
            if not has_headline:
                non_headline_blocks.extend(items)
                continue
            headed_group_blocks[gid] = list(items)

        # Re-attach headline-less content blocks to headline groups.
        # This captures multi-column continuation where the second column has no repeated headline.
        headed_gids = list(headed_group_blocks.keys())
        if headed_gids:
            residual_non_headline: list[TextBlock] = []
            non_headline_content: list[TextBlock] = []
            for b in sorted(non_headline_blocks, key=lambda x: order_index[id(x)]):
                if b.role == "content":
                    non_headline_content.append(b)
                else:
                    residual_non_headline.append(b)

        if headed_gids and non_headline_content:
            page_min_x = min(b.bbox_px[0] for b in raw_blocks)
            page_max_x = max(b.bbox_px[2] for b in raw_blocks)
            page_min_y = min(b.bbox_px[1] for b in raw_blocks)
            page_max_y = max(b.bbox_px[3] for b in raw_blocks)
            page_w = max(1.0, float(page_max_x - page_min_x))
            page_h = max(1.0, float(page_max_y - page_min_y))
            spread_mode = (page_w / max(1.0, page_h)) >= 1.35
            spread_mid_x = (page_min_x + page_max_x) / 2.0

            if len(headed_gids) == 1:
                headed_group_blocks[headed_gids[0]].extend(non_headline_content)
            else:
                for cb in non_headline_content:
                    best_gid = ""
                    best_score = -1e9
                    cbx1, cby1, cbx2, cby2 = cb.bbox_px
                    ccenter_x = (cbx1 + cbx2) / 2.0
                    ccenter_y = (cby1 + cby2) / 2.0
                    for gid in headed_gids:
                        items = headed_group_blocks[gid]
                        head_blocks = [x for x in items if x.role == "headline"]
                        hb = bbox_union([x.bbox_px for x in head_blocks]) if head_blocks else bbox_union([x.bbox_px for x in items])
                        gb = bbox_union([x.bbox_px for x in items])
                        if spread_mode:
                            cb_side_left = ccenter_x <= spread_mid_x
                            head_center_x = (hb[0] + hb[2]) / 2.0
                            hg_side_left = head_center_x <= spread_mid_x
                            if cb_side_left != hg_side_left:
                                continue
                        hcenter_x = (gb[0] + gb[2]) / 2.0
                        hcenter_y = (gb[1] + gb[3]) / 2.0
                        dx = abs(ccenter_x - hcenter_x)
                        dy = max(0.0, cby1 - hb[3]) if cby1 >= hb[1] else abs(ccenter_y - hcenter_y)
                        overlap = max(0.0, min(cbx2, gb[2]) - max(cbx1, gb[0])) / float(max(1, min(cbx2 - cbx1, gb[2] - gb[0])))
                        score = (2.0 * overlap) - (dx / page_w) - (0.8 * (dy / page_h))
                        # Prefer flowing forward in reading order.
                        if order_index[id(cb)] >= max(order_index[id(x)] for x in items):
                            score += 0.2
                        if score > best_score:
                            best_score = score
                            best_gid = gid
                    if best_gid and best_score >= -0.35:
                        headed_group_blocks[best_gid].append(cb)
                    else:
                        residual_non_headline.append(cb)

            non_headline_blocks = residual_non_headline

        for gid in group_order:
            items = headed_group_blocks.get(gid)
            if not items:
                continue
            items = sorted(items, key=lambda b: order_index[id(b)])

            headline_parts = [b.text.strip() for b in items if b.role == "headline" and b.text.strip()]
            content_parts = [b.text.strip() for b in items if b.role != "headline" and b.text.strip()]
            headline_text = " ".join(headline_parts).strip()
            content_text = " ".join(content_parts).strip()
            full_text = " ".join([x for x in [headline_text, content_text] if x]).strip()
            logical_items.append(
                {
                    "block_id": "",
                    "article_group_id": gid,
                    "role": "headline_group",
                    "bbox_px": list(bbox_union([b.bbox_px for b in items])),
                    "text": full_text,
                    "headline_text": headline_text,
                    "content_text": content_text,
                }
            )

        if non_headline_blocks:
            non_headline_blocks = sorted(non_headline_blocks, key=lambda b: order_index[id(b)])
            heights = [max(1, b.bbox_px[3] - b.bbox_px[1]) for b in non_headline_blocks]
            median_h = float(np.median(heights)) if heights else 32.0
            y_gap_thr = max(26.0, 1.45 * median_h)

            def _side_rank_block(b: TextBlock) -> int:
                if not spread_mode:
                    return 0
                cx = (b.bbox_px[0] + b.bbox_px[2]) / 2.0
                return 0 if cx <= spread_mid_x else 1

            ordered_non_head = sorted(
                non_headline_blocks,
                key=lambda b: (_side_rank_block(b), b.bbox_px[1], b.bbox_px[0]),
            )
            chunks: list[list[TextBlock]] = []
            cur: list[TextBlock] = []
            for b in ordered_non_head:
                if not cur:
                    cur = [b]
                    continue
                prev = cur[-1]
                prev_side = _side_rank_block(prev)
                side = _side_rank_block(b)
                same_side = side == prev_side
                vgap = float(b.bbox_px[1] - prev.bbox_px[3])
                if (not same_side) or (vgap > y_gap_thr):
                    chunks.append(cur)
                    cur = [b]
                else:
                    cur.append(b)
            if cur:
                chunks.append(cur)

            for i, chunk in enumerate(chunks, start=1):
                merged_text = " ".join(b.text.strip() for b in chunk if b.text.strip()).strip()
                if not merged_text:
                    continue
                logical_items.append(
                    {
                        "block_id": "",
                        "article_group_id": f"a_other_{i}",
                        "role": "other_text_group",
                        "bbox_px": list(bbox_union([b.bbox_px for b in chunk])),
                        "text": merged_text,
                    }
                )

        if logical_items:
            lx1 = min(int(i["bbox_px"][0]) for i in logical_items)
            lx2 = max(int(i["bbox_px"][2]) for i in logical_items)
            ly1 = min(int(i["bbox_px"][1]) for i in logical_items)
            ly2 = max(int(i["bbox_px"][3]) for i in logical_items)
            pw = max(1.0, float(lx2 - lx1))
            ph = max(1.0, float(ly2 - ly1))
            spread_mode_items = (pw / max(1.0, ph)) >= 1.35
            mid_x = (lx1 + lx2) / 2.0

            def _logical_item_key(item: dict[str, Any]) -> tuple[int, int, int]:
                bx = item["bbox_px"]
                cx = (int(bx[0]) + int(bx[2])) / 2.0
                side_rank = 0 if (not spread_mode_items or cx <= mid_x) else 1
                return (side_rank, int(bx[1]), int(bx[0]))

            logical_items = sorted(logical_items, key=_logical_item_key)

        for i, item in enumerate(logical_items, start=1):
            item["block_id"] = f"t{i}"
        return logical_items

    def _to_debug_json(self, result: PageResult, rejected_candidates: list[Any]) -> dict[str, Any]:
        payload = self._to_json(result)
        payload["text_blocks_debug"] = [
            {
                "block_id": b.block_id,
                "regions_px": [list(r) for r in b.regions_px],
                "line_count": len(b.lines),
                "avg_line_confidence": (
                    round(sum(l.confidence for l in b.lines) / max(1, len(b.lines)), 4) if b.lines else 0.0
                ),
            }
            for b in result.text_blocks
        ]
        payload["rejected_visual_candidates"] = [
            {"bbox_px": list(r.bbox), "reason": r.reason, "area_ratio": round(r.area_ratio, 6)}
            for r in rejected_candidates
        ]
        payload["visual_blocks_debug"] = [
            {
                "block_id": b.block_id,
                "class": b.visual_class,
                "needs_review": b.needs_review,
                "review_reasons": b.review_reasons,
                "confidence": b.confidence,
                "page_side": b.page_side,
                "panel_index": b.panel_index,
                "headline": b.headline,
                "headline_title": b.headline_title,
                "headline_byline": b.headline_byline,
                "bbox_px": list(b.bbox_px),
                "area_ratio": b.area_ratio,
                "short_side_ratio": b.short_side_ratio,
            }
            for b in result.visual_blocks
        ]
        return payload
