from __future__ import annotations

import statistics
import re

from src.config import ClassificationConfig
from src.pipeline.models import TextBlock
from src.utils.geometry import bbox_union, horizontal_overlap_ratio


def _line_gaps(lines: list) -> list[float]:
    if len(lines) < 2:
        return []
    out: list[float] = []
    for i in range(len(lines) - 1):
        out.append(max(0.0, float(lines[i + 1].bbox_px[1] - lines[i].bbox_px[3])))
    return out


def _looks_like_title_pattern(first_text: str, rest_chars: int, cfg: ClassificationConfig) -> bool:
    txt = re.sub(r"\s+", " ", first_text.strip())
    if not txt:
        return False
    if len(txt) > cfg.prefix_title_pattern_max_chars:
        return False
    if rest_chars < cfg.prefix_title_pattern_min_rest_chars:
        return False
    alpha_count = len(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]", txt))
    if alpha_count < cfg.headline_min_alpha_chars:
        return False
    has_delim = ("/" in txt) or (":" in txt)
    upper_tokens = [t for t in re.split(r"\s+", txt) if t.isupper() and len(t) >= 3]
    return has_delim or len(upper_tokens) >= 1


def _headline_text_pattern(clean_text: str) -> bool:
    txt = re.sub(r"\s+", " ", clean_text.strip())
    if not txt:
        return False
    words = [w for w in txt.split(" ") if w]
    if len(words) < 2 or len(words) > 10:
        return False
    if txt.endswith((".", "!", "?")):
        return False
    if ":" in txt and len(words) <= 4:
        return False
    if "/" in txt:
        return True
    punct = len(re.findall(r"[,:;]", txt))
    if punct > 1:
        return False
    return True


def _looks_like_dialogue_or_bubble(clean_text: str) -> bool:
    txt = re.sub(r"\s+", " ", clean_text.strip())
    if not txt:
        return False
    if re.match(r"^[-–—]\s*\w+", txt):
        return True
    if txt.startswith(("“", "\"", "'", "‘")) and txt.endswith(("”", "\"", "'", "’")):
        return True
    # Dialogue-like snippets are usually short and punctuated as spoken lines.
    if len(txt) <= 90 and txt.endswith(("...", ".", "!", "?")) and ("," in txt or "..." in txt):
        return True
    return False


def _reasonable_headline_text(clean_text: str) -> bool:
    txt = re.sub(r"\s+", " ", clean_text.strip())
    if not txt:
        return False
    first_alpha_match = re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]", txt)
    if first_alpha_match:
        ch = first_alpha_match.group(0)
        if ch.lower() == ch:
            return False
    words = [w for w in txt.split(" ") if w]
    if len(words) >= 2:
        return True
    # Allow single-token headline only for acronym-like labels.
    token = words[0] if words else ""
    return bool(re.fullmatch(r"[A-ZÇĞİÖŞÜ0-9]{3,}", token))


def _split_prefix_headline_blocks(blocks: list[TextBlock], cfg: ClassificationConfig) -> list[TextBlock]:
    split_blocks: list[TextBlock] = []
    for block in blocks:
        lines = sorted(block.lines, key=lambda l: (l.bbox_px[1], l.bbox_px[0]))
        if len(lines) < cfg.prefix_split_min_total_lines:
            split_blocks.append(block)
            continue

        first = lines[0]
        rest = lines[1:]
        rest_heights = [l.line_height for l in rest]
        rest_median_h = statistics.median(rest_heights) if rest_heights else first.line_height
        first_h = float(first.line_height)
        first_w = float(first.line_width)
        rest_w_median = statistics.median([l.line_width for l in rest]) if rest else first_w
        first_text = first.text.strip()
        rest_text = " ".join(l.text for l in rest).strip()
        rest_chars = len(re.sub(r"\s+", " ", rest_text))

        gaps = _line_gaps(lines)
        first_gap = gaps[0] if gaps else 0.0
        rest_gap_med = statistics.median(gaps[1:]) if len(gaps) > 2 else (gaps[0] if gaps else 0.0)
        gap_ok = (
            first_gap >= rest_gap_med * cfg.prefix_headline_gap_ratio
            or first_gap >= rest_median_h * cfg.prefix_headline_gap_height_ratio
        )
        size_ok = first_h >= rest_median_h * cfg.prefix_headline_height_ratio
        width_ok = first_w <= rest_w_median * cfg.prefix_headline_width_ratio_max
        pattern_ok = _looks_like_title_pattern(first_text, rest_chars, cfg)

        should_split = (size_ok and (gap_ok or width_ok)) or pattern_ok
        if not should_split:
            split_blocks.append(block)
            continue

        headline_regions = [first.bbox_px]
        content_regions = [l.bbox_px for l in rest]
        headline_block = TextBlock(
            block_id=block.block_id,
            article_group_id="",
            role="other_text",
            bbox_px=bbox_union(headline_regions),
            regions_px=headline_regions,
            text=first_text,
            lines=[first],
        )
        content_block = TextBlock(
            block_id=block.block_id,
            article_group_id="",
            role="other_text",
            bbox_px=bbox_union(content_regions),
            regions_px=content_regions,
            text=rest_text,
            lines=rest,
        )
        split_blocks.extend([headline_block, content_block])
    return split_blocks


def classify_text_blocks(blocks: list[TextBlock], cfg: ClassificationConfig) -> None:
    if not blocks:
        return

    blocks[:] = _split_prefix_headline_blocks(blocks, cfg)

    heights = [line.line_height for b in blocks for line in b.lines]
    page_median_h = statistics.median(heights) if heights else 1.0
    page_h = max(1, max(b.bbox_px[3] for b in blocks) - min(b.bbox_px[1] for b in blocks))

    stats: list[dict[str, float | int | bool]] = []
    for block in blocks:
        line_count = len(block.lines)
        clean_text = block.text.replace("\n", " ").strip()
        chars = len(clean_text)
        digit_count = len(re.findall(r"\d", clean_text))
        alpha_count = len(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]", clean_text))
        digit_ratio = digit_count / max(1, chars)
        block_median_h = statistics.median([l.line_height for l in block.lines]) if block.lines else page_median_h
        avg_conf = (
            sum(float(getattr(l, "confidence", 0.0)) for l in block.lines) / max(1, len(block.lines))
            if block.lines
            else 0.0
        )

        headline_like = (
            block_median_h >= page_median_h * cfg.headline_height_ratio
            and line_count <= cfg.headline_max_lines
            and chars <= cfg.headline_max_chars
            and digit_ratio <= cfg.headline_max_digit_ratio
            and alpha_count >= cfg.headline_min_alpha_chars
        )
        content_like = (
            line_count >= cfg.content_min_lines
            or (
                block_median_h <= page_median_h * cfg.content_height_upper_ratio
                and chars >= cfg.content_min_chars
            )
        )
        stats.append(
            {
                "headline_like": headline_like,
                "content_like": content_like,
                "line_count": line_count,
                "chars": chars,
                "digit_ratio": digit_ratio,
                "alpha_count": alpha_count,
                "clean_text": clean_text,
                "avg_conf": avg_conf,
            }
        )

    content_blocks = [b for i, b in enumerate(blocks) if bool(stats[i]["content_like"])]

    for i, block in enumerate(blocks):
        headline_like = bool(stats[i]["headline_like"])
        content_like = bool(stats[i]["content_like"])
        clean_text = str(stats[i]["clean_text"])
        dialogue_like = _looks_like_dialogue_or_bubble(clean_text)

        has_near_content = False
        near_content_width = 0.0
        if headline_like:
            for content in content_blocks:
                if content is block:
                    continue
                if content.bbox_px[1] < block.bbox_px[1]:
                    continue
                v_gap = max(0, content.bbox_px[1] - block.bbox_px[3])
                if v_gap > page_h * cfg.pair_max_vertical_ratio:
                    continue
                h_overlap = horizontal_overlap_ratio(block.bbox_px, content.bbox_px)
                if h_overlap >= cfg.headline_content_min_overlap:
                    has_near_content = True
                    near_content_width = max(near_content_width, float(content.bbox_px[2] - content.bbox_px[0]))
                    break
        else:
            for content in content_blocks:
                if content is block:
                    continue
                if content.bbox_px[1] < block.bbox_px[1]:
                    continue
                v_gap = max(0, content.bbox_px[1] - block.bbox_px[3])
                if v_gap > page_h * cfg.pair_max_vertical_ratio:
                    continue
                h_overlap = horizontal_overlap_ratio(block.bbox_px, content.bbox_px)
                if h_overlap >= cfg.headline_content_min_overlap:
                    has_near_content = True
                    near_content_width = max(near_content_width, float(content.bbox_px[2] - content.bbox_px[0]))
                    break

        short_title_like = (
            int(stats[i]["line_count"]) <= 2
            and int(stats[i]["chars"]) <= cfg.headline_max_chars
            and float(stats[i]["digit_ratio"]) <= cfg.headline_max_digit_ratio
            and int(stats[i]["alpha_count"]) >= cfg.headline_min_alpha_chars
            and float(stats[i]["avg_conf"]) >= 0.55
            and _headline_text_pattern(str(stats[i]["clean_text"]))
        )
        block_w = float(block.bbox_px[2] - block.bbox_px[0])
        narrower_than_content = near_content_width > 0 and block_w <= near_content_width * 0.92
        strong_headline_like = (
            int(stats[i]["line_count"]) <= 2
            and int(stats[i]["chars"]) <= min(90, cfg.headline_max_chars)
            and int(stats[i]["alpha_count"]) >= cfg.headline_min_alpha_chars
            and float(stats[i]["avg_conf"]) >= 0.58
            and not dialogue_like
            and headline_like
        )

        headline_text_ok = _reasonable_headline_text(clean_text) and _headline_text_pattern(clean_text)

        if (
            not dialogue_like
            and not content_like
            and has_near_content
            and headline_text_ok
            and (
                ((headline_like or short_title_like) and narrower_than_content)
                or strong_headline_like
            )
        ):
            block.role = "headline"
        elif content_like:
            block.role = "content"
        else:
            block.role = "other_text"
