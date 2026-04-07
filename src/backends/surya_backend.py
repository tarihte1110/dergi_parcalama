from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from src.backends.ocr_base import OCRBackend, OCRLine
from src.config import OCRConfig
from src.utils.geometry import BBox, bbox_center


@dataclass
class _SuryaRuntime:
    det_predictor: Any
    rec_predictor: Any
    task_name: str


class SuryaBackend(OCRBackend):
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        try:
            import torch
            from surya.common.surya.schema import TaskNames
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise ImportError("surya-ocr is not installed. Run `pip install -r requirements.txt`.") from exc

        requested = (config.surya_device or "mps").lower()
        if requested == "mps" and torch.backends.mps.is_available():
            device = "mps"
        elif requested in {"cpu", "mps"}:
            device = "cpu" if requested == "mps" else requested
        else:
            device = requested

        foundation_predictor = FoundationPredictor(device=device)
        det_predictor = DetectionPredictor(device=device)
        rec_predictor = RecognitionPredictor(foundation_predictor)

        self.runtime = _SuryaRuntime(
            det_predictor=det_predictor,
            rec_predictor=rec_predictor,
            task_name=str(TaskNames.ocr_with_boxes),
        )

    @staticmethod
    def _polygon_to_bbox(polygon: list[list[float]]) -> BBox:
        xs = [float(pt[0]) for pt in polygon]
        ys = [float(pt[1]) for pt in polygon]
        x1 = int(round(min(xs)))
        y1 = int(round(min(ys)))
        x2 = int(round(max(xs)))
        y2 = int(round(max(ys)))
        return x1, y1, x2, y2

    def detect(self, image: np.ndarray) -> list[OCRLine]:
        image_rgb = image[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        w, h = pil.size
        long_edge = max(w, h)
        highres_scale = max(1.0, float(self.config.surya_highres_scale))
        target_long = min(
            int(round(long_edge * highres_scale)),
            int(self.config.surya_highres_max_long_edge),
        )
        if target_long > long_edge:
            ratio = target_long / float(long_edge)
            highres = pil.resize((int(round(w * ratio)), int(round(h * ratio))), Image.Resampling.LANCZOS)
        else:
            highres = pil

        preds = self.runtime.rec_predictor(
            [pil],
            task_names=[self.runtime.task_name],
            det_predictor=self.runtime.det_predictor,
            detection_batch_size=self.config.surya_detection_batch_size,
            recognition_batch_size=self.config.surya_recognition_batch_size,
            highres_images=[highres],
            math_mode=False,
            return_words=False,
        )

        lines: list[OCRLine] = []
        if not preds:
            return lines

        first = preds[0]
        for item in first.text_lines:
            text = str(item.text).strip()
            conf = float(item.confidence)
            if not text or conf < self.config.min_confidence:
                continue

            polygon = np.asarray(item.polygon).tolist()
            bbox = self._polygon_to_bbox(polygon)
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

        lines.sort(key=lambda l: (l.bbox_px[1], l.bbox_px[0]))
        return lines
