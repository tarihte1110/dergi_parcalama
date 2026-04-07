from __future__ import annotations

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium


def iter_pdf_pages_bgr(pdf_path: Path, scale: float = 2.0) -> list[tuple[int, np.ndarray]]:
    doc = pdfium.PdfDocument(str(pdf_path))
    pages: list[tuple[int, np.ndarray]] = []
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()
        rgb = np.array(pil)
        bgr = rgb[:, :, ::-1].copy()
        pages.append((i + 1, bgr))
    return pages
