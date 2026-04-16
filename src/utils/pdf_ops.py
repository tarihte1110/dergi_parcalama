from __future__ import annotations

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium


def iter_pdf_pages_bgr_stream(pdf_path: Path, scale: float = 2.0):
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(doc)):
            page = doc[i]
            pil = None
            rgb = None
            try:
                pil = page.render(scale=scale).to_pil()
                rgb = np.array(pil)
                bgr = rgb[:, :, ::-1].copy()
                yield (i + 1, bgr)
            finally:
                if pil is not None:
                    pil.close()
                del rgb
                page.close()
    finally:
        doc.close()


def iter_pdf_pages_bgr(pdf_path: Path, scale: float = 2.0) -> list[tuple[int, np.ndarray]]:
    return list(iter_pdf_pages_bgr_stream(pdf_path=pdf_path, scale=scale))
