from src.backends.ocr_base import OCRBackend, OCRLine
from src.backends.paddle_ocr_backend import PaddleOCRBackend
from src.backends.surya_backend import SuryaBackend

__all__ = ["OCRBackend", "OCRLine", "PaddleOCRBackend", "SuryaBackend"]
