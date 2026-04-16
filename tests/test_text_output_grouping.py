from __future__ import annotations

import numpy as np

from src.backends.ocr_base import OCRBackend, OCRLine
from src.config import AppConfig
from src.pipeline.models import TextBlock
from src.pipeline.page_pipeline import DocumentPipeline
from src.utils.logging_utils import setup_logger


class EmptyBackend(OCRBackend):
    def detect(self, image: np.ndarray) -> list[OCRLine]:
        return []


def test_compose_logical_text_blocks_merges_headline_group_and_other() -> None:
    pipeline = DocumentPipeline(config=AppConfig(), logger=setup_logger("test_grouping"), backend=EmptyBackend())
    raw = [
        TextBlock("t1", "a1", "headline", (10, 10, 110, 30), [(10, 10, 110, 30)], "BASLIK", []),
        TextBlock("t2", "a1", "content", (10, 40, 180, 120), [(10, 40, 180, 120)], "icerik 1", []),
        TextBlock("t3", "a2", "content", (220, 20, 300, 70), [(220, 20, 300, 70)], "diger 1", []),
        TextBlock("t4", "a3", "other_text", (220, 90, 330, 150), [(220, 90, 330, 150)], "diger 2", []),
    ]
    out = pipeline._compose_logical_text_blocks(raw)
    assert len(out) == 2
    assert out[0]["role"] == "headline_group"
    assert "BASLIK" in out[0]["headline_text"]
    assert "icerik 1" in out[0]["content_text"]
    assert "diger 1" in out[0]["content_text"]
    assert out[1]["role"] == "other_text_group"
    assert "diger 2" in out[1]["text"]
