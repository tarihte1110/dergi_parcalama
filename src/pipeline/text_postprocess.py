from __future__ import annotations

import re
import unicodedata


def clean_ocr_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Join accidental single-letter splits in Turkish/Latin words.
    s = re.sub(r"\b([A-Za-zÇĞİÖŞÜçğıöşü])\s+([a-zçğıöşü])", r"\1\2", s)

    # Punctuation spacing normalization.
    s = s.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    s = s.replace(" ;", ";").replace(" :", ":")

    # Common OCR spacing around apostrophe/percent.
    s = re.sub(r"\s+'\s*", "'", s)
    s = re.sub(r"%\s+(\d)", r"%\1", s)

    # Remove obvious duplicated adjacent words (case-insensitive).
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s, flags=re.IGNORECASE)

    # Hyphenation leftovers.
    s = s.replace("- ", "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s
