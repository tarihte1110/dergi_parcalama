from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.backends.ocr_base import OCRBackend, OCRLine
from src.config import OCRConfig
from src.utils.geometry import BBox, bbox_center


class PaddleOCRBackend(OCRBackend):
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise ImportError("paddleocr is not installed. Run `pip install -r requirements.txt`.") from exc

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self._ocr = PaddleOCR(
            lang=config.paddle_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=bool(config.paddle_use_angle_cls),
        )

    @staticmethod
    def _quad_to_bbox(quad: list[list[float]]) -> BBox:
        xs = [float(pt[0]) for pt in quad]
        ys = [float(pt[1]) for pt in quad]
        x1, x2 = int(round(min(xs))), int(round(max(xs)))
        y1, y2 = int(round(min(ys))), int(round(max(ys)))
        return x1, y1, x2, y2

    def _parse_v3_result(self, result: Any) -> list[OCRLine]:
        lines: list[OCRLine] = []
        items = list(result) if isinstance(result, list) else [result]
        for page in items:
            page_dict = page if isinstance(page, dict) else dict(page)
            polys = page_dict.get("dt_polys") or page_dict.get("rec_polys") or []
            texts = page_dict.get("rec_texts") or []
            scores = page_dict.get("rec_scores") or []
            for quad, text, conf in zip(polys, texts, scores):
                text_s = str(text).strip()
                conf_f = float(conf)
                if not text_s or conf_f < self.config.min_confidence:
                    continue
                quad_list = np.asarray(quad).tolist()
                bbox = self._quad_to_bbox(quad_list)
                width = max(1, bbox[2] - bbox[0])
                height = max(1, bbox[3] - bbox[1])
                lines.append(
                    OCRLine(
                        text=text_s,
                        confidence=conf_f,
                        bbox_px=bbox,
                        line_height=float(height),
                        line_width=float(width),
                        center=bbox_center(bbox),
                    )
                )
        return lines

    def _parse_v2_result(self, result: list[Any]) -> list[OCRLine]:
        lines: list[OCRLine] = []
        if not result:
            return lines
        for page in result:
            if page is None:
                continue
            for item in page:
                if not item or len(item) < 2:
                    continue
                quad = item[0]
                text_info = item[1]
                if not text_info or len(text_info) < 2:
                    continue
                text = str(text_info[0]).strip()
                conf = float(text_info[1])
                if not text or conf < self.config.min_confidence:
                    continue
                bbox = self._quad_to_bbox(quad)
                width = max(1, bbox[2] - bbox[0])
                height = max(1, bbox[3] - bbox[1])
                lines.append(
                    OCRLine(
                        text=text,
                        confidence=conf,
                        bbox_px=bbox,
                        line_height=float(height),
                        line_width=float(width),
                        center=bbox_center(bbox),
                    )
                )
        return lines

    def detect(self, image: np.ndarray) -> list[OCRLine]:
        lines: list[OCRLine]
        if hasattr(self._ocr, "predict"):
            result = self._ocr.predict(image)
            lines = self._parse_v3_result(result)
        else:
            result = self._ocr.ocr(image, cls=self.config.paddle_use_angle_cls)
            lines = self._parse_v2_result(result)

        lines.sort(key=lambda l: (l.bbox_px[1], l.bbox_px[0]))
        return lines
