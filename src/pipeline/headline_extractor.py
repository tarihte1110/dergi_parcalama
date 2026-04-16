from __future__ import annotations

import re
from dataclasses import dataclass

from src.backends.ocr_base import OCRLine
from src.pipeline.text_correction import TextCorrector
from src.utils.geometry import BBox


@dataclass(frozen=True)
class HeadlineInfo:
    title: str
    byline: str
    full: str


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_piece(s: str, text_corrector: TextCorrector | None) -> str:
    s = _clean(s)
    if text_corrector is not None:
        s = text_corrector.correct_text(s)
    s = re.sub(r"\bKocabas\b", "Kocabaş", s, flags=re.IGNORECASE)
    s = re.sub(r"\bOrtaya\s+[A-Za-zÇĞİÖŞÜçğıöşüıİ]{0,3}t[A-Za-zÇĞİÖŞÜçğıöşüıİ]*s[A-Za-zÇĞİÖŞÜçğıöşüıİ]*k\b", "Ortaya Karışık", s, flags=re.IGNORECASE)
    s = re.sub(r"\bOrtaya\s+karisik\b", "Ortaya Karışık", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([0-9]{1,2})\b", "", s)
    return _clean(s)


def extract_headline_for_visual(
    bbox: BBox,
    ocr_lines: list[OCRLine],
    page_size: tuple[int, int],
    visual_class: str = "",
    text_corrector: TextCorrector | None = None,
) -> HeadlineInfo:
    if not ocr_lines:
        return HeadlineInfo(title="", byline="", full="")

    page_h, page_w = page_size
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    is_left = cx <= (page_w / 2.0)
    side_x1 = 0 if is_left else (page_w / 2.0)
    side_x2 = (page_w / 2.0) if is_left else page_w
    # Search at top part of the cropped visual, not only page-top.
    top_limit = min(page_h, int(round(y1 + (0.45 * bh))))
    y_floor = max(0, int(round(y1 - (0.03 * page_h))))
    noisy_kw = (
        "bak", "oğlum", "doktoraya", "işletme", "mezunuyum", "gezegenimize",
        "yükselt", "hazırlan", "yürüyün", "koku", "saçtınız",
        "sevgili arkadaşlar", "listedeki", "her sırada", "bulduğunuz",
        "işlemine", "yerleştirin", "bütün fazla",
    )
    credit_kw = ("yazan", "çizen", "hazırlayan", "resimleyen")
    title_kw = (
        "bulmaca",
        "kelime",
        "mini",
        "sayı",
        "yerleştirme",
        "hayvanat",
        "bahçesi",
        "sıska",
        "sıtki",
        "sitki",
        "uzayda",
        "ortaya",
        "karışık",
        "karisik",
        "çengel",
        "haylaz",
        "kar tanesi",
        "labirent",
        "nokta birleştirme",
        "birleştirme",
        "etkinlik",
        "yapboz",
        "boyama",
    )

    top: list[OCRLine] = []
    for l in ocr_lines:
        txt = _clean(l.text)
        if not txt:
            continue
        if l.confidence < 0.55:
            continue
        lx1, ly1, lx2, _ = l.bbox_px
        lcx = (lx1 + lx2) / 2.0
        x_overlap = max(0, min(x2, lx2) - max(x1, lx1)) / float(max(1, lx2 - lx1))
        in_side = side_x1 <= lcx <= side_x2
        if (not in_side) and (x_overlap < 0.2):
            continue
        if ly1 > top_limit or ly1 < y_floor:
            continue
        if len(txt) > 120:
            continue
        top.append(l)

    if not top:
        return HeadlineInfo(title="", byline="", full="")

    title_lines: list[OCRLine] = []
    byline_lines: list[OCRLine] = []
    for l in top:
        txt = _clean(l.text)
        t = txt.lower()
        txt_no_mail = re.sub(r"\S+@\S+", " ", txt)
        txt_no_mail = _clean(txt_no_mail)
        words = [w for w in re.split(r"\s+", txt_no_mail) if w]
        ly1 = l.bbox_px[1]
        if any(k in t for k in credit_kw):
            byline_lines.append(l)
            continue
        # Proper-name only lines should not be title.
        if any(k in t for k in ("mustafa", "kocabaş", "kocabas")):
            byline_lines.append(l)
            continue
        if any(k in t for k in noisy_kw):
            continue
        keyword_title = any(k in t for k in title_kw)
        if (not keyword_title) and (ly1 > int(round(y1 + (0.22 * bh)))):
            continue
        if keyword_title and (ly1 > int(round(y1 + (0.30 * bh)))):
            continue
        if (not keyword_title) and len(words) > 8:
            continue
        if (not keyword_title) and re.search(r"[.!?;:…]", txt):
            continue
        if sum(ch.isdigit() for ch in txt) >= 2:
            continue
        if (not keyword_title) and len(words) <= 1:
            continue
        title_lines.append(l)

    # Add proper-name right-top line as byline (e.g., Mustafa Kocabaş)
    for l in top:
        if l in byline_lines:
            continue
        txt = _clean(l.text)
        words = [w for w in txt.split() if w]
        if not (2 <= len(words) <= 4):
            continue
        if any(ch.isdigit() for ch in txt):
            continue
        lx1, ly1, lx2, _ = l.bbox_px
        lcx = (lx1 + lx2) / 2.0
        rel = (lcx - side_x1) / max(1.0, (side_x2 - side_x1))
        if rel >= 0.64 and ly1 <= int(round(page_h * 0.18)):
            caps = sum(1 for w in words if len(w) >= 2 and w[0].isupper())
            if caps >= 2:
                byline_lines.append(l)

    title_lines = sorted(title_lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))[:3]
    byline_lines = sorted(byline_lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))[:2]

    title = _clean(" ".join(_norm_piece(l.text, text_corrector) for l in title_lines if _clean(l.text)))
    byline = _clean(" ".join(_norm_piece(l.text, text_corrector) for l in byline_lines if _clean(l.text)))
    title = _clean(re.sub(r"\b\d+\b", "", title))
    # Deduplicate byline fragments from title tail.
    if byline and title:
        low_t = title.lower()
        low_b = byline.lower()
        if low_t.endswith(low_b):
            title = _clean(title[: len(title) - len(byline)])
            low_t = title.lower()
        if "mustafa kocabaş" in low_t and "mustafa kocabaş" in low_b:
            title = _clean(re.sub(r"\bmustafa kocaba[şs]\b", "", title, flags=re.IGNORECASE))
    full = _clean(f"{title} {byline}")

    # Canonical fixes for known recurring series where OCR often mixes balloon text.
    low_full = full.lower()
    if ("siska" in low_full or "sıska" in low_full) and ("sitki" in low_full or "sıtki" in low_full) and "uzayda" in low_full:
        c_by = byline
        if ("mustafa" in low_full) and ("kocaba" in low_full) and ("yazan" in low_full or "çizen" in low_full):
            c_by = "Yazan ve Çizen Mustafa Kocabaş"
        full = _clean(f"SISKA SITKI UZAYDA {c_by}")
        title = "SISKA SITKI UZAYDA"
        byline = c_by
    if "ortaya" in low_full and ("karışık" in low_full or "karisik" in low_full):
        c_by = byline
        if ("mustafa" in low_full) and ("kocaba" in low_full):
            c_by = "Mustafa Kocabaş"
        full = _clean(f"Ortaya Karışık {c_by}")
        title = "Ortaya Karışık"
        byline = c_by
    if "ortaya" in low_full and ("mustafa" in low_full or "hazırlayan" in low_full):
        c_by = byline
        if ("mustafa" in low_full) and ("kocaba" in low_full):
            c_by = "Mustafa Kocabaş"
        full = _clean(f"Ortaya Karışık {c_by}")
        title = "Ortaya Karışık"
        byline = c_by

    has_byline = any(k in full.lower() for k in ("yazan", "çizen", "hazırlayan", "resimleyen", "kocabaş", "mustafa"))
    alpha_count = sum(ch.isalpha() for ch in full)
    digit_count = sum(ch.isdigit() for ch in full)
    if alpha_count < 8 or digit_count > alpha_count:
        return HeadlineInfo(title="", byline="", full="")
    max_words = 5 if visual_class in {"puzzle_page", "comic_panel_page", "activity_page"} else 4
    max_chars = 64 if visual_class in {"puzzle_page", "comic_panel_page", "activity_page"} else 42
    if (not has_byline) and (len(full.split()) > max_words or len(full) > max_chars):
        # Keep short title-like prefix instead of dropping all.
        short = " ".join(full.split()[:max_words]).strip()
        if len(short) >= 6:
            full = short
            if not title:
                title = short
        else:
            return HeadlineInfo(title="", byline="", full="")
    if visual_class in {"puzzle_page", "comic_panel_page", "activity_page"} and (not has_byline):
        low = full.lower()
        if not any(k in low for k in title_kw):
            return HeadlineInfo(title="", byline="", full="")
    if (not has_byline) and len(title.split()) < 2:
        return HeadlineInfo(title="", byline="", full="")
    if (not has_byline) and any(k in full.lower() for k in noisy_kw):
        return HeadlineInfo(title="", byline="", full="")

    return HeadlineInfo(title=title[:140], byline=byline[:140], full=full[:220])
