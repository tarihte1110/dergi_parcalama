from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

import cv2

from src.backends.ocr_base import OCRBackend, OCRLine
from src.backends.paddle_ocr_backend import PaddleOCRBackend
from src.backends.surya_backend import SuryaBackend
from src.config import AppConfig
from src.pipeline.article_grouping import assign_article_groups
from src.pipeline.cropper import save_visual_crop
from src.pipeline.debug_viz import save_debug_views
from src.pipeline.models import PageResult, TextBlock, VisualBlock
from src.pipeline.non_text_mask import build_non_text_mask
from src.pipeline.reading_order import order_blocks_for_reading
from src.pipeline.text_classification import classify_text_blocks
from src.pipeline.text_correction import TextCorrector
from src.pipeline.text_detection import detect_text_lines
from src.pipeline.text_grouping import group_text_lines
from src.pipeline.text_mask import build_text_mask
from src.pipeline.text_postprocess import clean_ocr_text
from src.pipeline.visual_candidates import build_visual_decisions, extract_visual_candidates
from src.utils.geometry import iou
from src.utils.io import prepare_output_dirs, write_json
from src.utils.image_ops import read_image


class DocumentPipeline:
    def __init__(self, config: AppConfig, logger: logging.Logger, backend: OCRBackend | None = None) -> None:
        self.config = config
        self.logger = logger
        self.backend = backend or self._build_backend()
        self.text_corrector = TextCorrector.from_config(config.text_correction)

    def _build_backend(self) -> OCRBackend:
        backend_name = self.config.ocr.backend.lower().strip()
        if backend_name == "surya":
            return SuryaBackend(self.config.ocr)
        if backend_name == "paddle":
            return PaddleOCRBackend(self.config.ocr)
        raise ValueError(f"Unsupported OCR backend: {backend_name}")

    def process_image(self, image_path: Path, output_root: Path, debug_override: bool | None = None) -> dict[str, Any]:
        image = read_image(image_path)
        return self.process_image_array(
            image=image,
            output_root=output_root,
            source_ref=str(image_path.as_posix()),
            stem=image_path.stem,
            debug_override=debug_override,
        )

    def process_image_array(
        self,
        image: Any,
        output_root: Path,
        source_ref: str,
        stem: str,
        debug_override: bool | None = None,
        ocr_lines: list[OCRLine] | None = None,
    ) -> dict[str, Any]:
        outputs = prepare_output_dirs(output_root)
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        page_ocr_lines = ocr_lines if ocr_lines is not None else detect_text_lines(image, self.backend)
        text_blocks = group_text_lines(page_ocr_lines, gray, self.config.grouping)
        classify_text_blocks(text_blocks, self.config.classification)
        assign_article_groups(text_blocks, h, self.config.classification)
        text_blocks = self._cleanup_text_blocks(text_blocks)
        text_blocks = order_blocks_for_reading(text_blocks, self.config.grouping)

        text_mask = build_text_mask((h, w), text_blocks, self.config.mask)
        non_text_mask = build_non_text_mask(image, text_mask, self.config.mask)
        candidates, rejected_candidates = extract_visual_candidates(image, non_text_mask, text_mask, self.config.visual)
        decisions = build_visual_decisions(candidates, image, non_text_mask, text_mask, self.config.visual)

        crops_page_dir = outputs["crops"] / stem
        crops_page_dir.mkdir(parents=True, exist_ok=True)

        visual_blocks: list[VisualBlock] = []
        for idx, d in enumerate(decisions, start=1):
            c = d.metrics
            crop_name = f"v{idx}.png"
            crop_path = crops_page_dir / crop_name
            padded_bbox = save_visual_crop(image, c.bbox, self.config.crop, crop_path)
            visual_blocks.append(
                VisualBlock(
                    block_id=f"v{idx}",
                    bbox_px=padded_bbox,
                    area_ratio=round(c.area_ratio, 6),
                    short_side_ratio=round(c.short_side_ratio, 6),
                    crop_path=str(crop_path.as_posix()),
                    visual_class=d.visual_class,
                    needs_review=d.needs_review,
                    review_reasons=d.review_reasons,
                )
            )

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

        payload = self._to_json(page_result)
        write_json(outputs["json"] / f"{stem}.json", payload)
        if self.config.runtime.write_debug_json:
            write_json(outputs["json_debug"] / f"{stem}.json", self._to_debug_json(page_result, rejected_candidates))

        do_debug = self.config.runtime.debug if debug_override is None else debug_override
        if do_debug:
            save_debug_views(
                outputs["debug"],
                stem,
                image,
                page_ocr_lines,
                text_blocks,
                text_mask,
                non_text_mask,
                candidates,
            )

        return payload

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
        text_items = []
        for b in result.text_blocks:
            text_items.append(
                {
                    "block_id": b.block_id,
                    "article_group_id": b.article_group_id,
                    "role": b.role,
                    "bbox_px": list(b.bbox_px),
                    "text": b.text,
                }
            )

        visual_items = []
        for b in result.visual_blocks:
            visual_items.append(
                {
                    "block_id": b.block_id,
                    "bbox_px": list(b.bbox_px),
                    "crop_path": b.crop_path,
                    "class": b.visual_class,
                    "needs_review": b.needs_review,
                }
            )

        return {
            "image_path": result.image_path,
            "page_size": {"width": result.page_width, "height": result.page_height},
            "text_blocks": text_items,
            "visual_blocks": visual_items,
        }

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
                "bbox_px": list(b.bbox_px),
                "area_ratio": b.area_ratio,
                "short_side_ratio": b.short_side_ratio,
            }
            for b in result.visual_blocks
        ]
        return payload
