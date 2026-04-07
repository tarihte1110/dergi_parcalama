from pathlib import Path

import cv2
import numpy as np

from src.backends.ocr_base import OCRBackend, OCRLine
from src.config import AppConfig
from src.pipeline import DocumentPipeline
from src.utils.io import prepare_output_dirs
from src.utils.logging_utils import setup_logger


class EmptyBackend(OCRBackend):
    def detect(self, image: np.ndarray) -> list[OCRLine]:
        return []


def test_pipeline_runs_with_empty_ocr(tmp_path: Path) -> None:
    image_path = tmp_path / "empty.png"
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), img)

    out_root = tmp_path / "outputs"
    prepare_output_dirs(out_root)

    pipeline = DocumentPipeline(config=AppConfig(), logger=setup_logger("test_logger"), backend=EmptyBackend())
    payload = pipeline.process_image(image_path=image_path, output_root=out_root, debug_override=False)

    assert payload["image_path"].endswith("empty.png")
    assert "text_blocks" in payload
    assert "visual_blocks" in payload
