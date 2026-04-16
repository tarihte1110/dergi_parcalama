from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pipeline outputs")
    p.add_argument("--output", type=Path, default=Path("outputs"), help="Output root")
    p.add_argument("--report", type=Path, default=None, help="Optional report path")
    return p.parse_args()


def _load_pages(output_root: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for result_path in sorted((output_root / "pages").glob("t_cocuk_p*/result.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["_page_id"] = result_path.parent.name
        pages.append(data)
    return pages


def build_report(pages: list[dict[str, Any]]) -> dict[str, Any]:
    visual_counts: list[int] = []
    visual_area: list[float] = []
    headline_counts: list[int] = []
    content_counts: list[int] = []
    other_counts: list[int] = []
    dialogue_headline: list[dict[str, str]] = []
    tiny_visuals: list[dict[str, Any]] = []
    huge_visuals: list[dict[str, Any]] = []

    for page in pages:
        page_id = page["_page_id"]
        w = int(page["page_size"]["width"])
        h = int(page["page_size"]["height"])

        vbs = page.get("visual_blocks", [])
        visual_counts.append(len(vbs))
        for vb in vbs:
            x1, y1, x2, y2 = vb["bbox_px"]
            ar = ((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
            visual_area.append(ar)
            if ar < 0.03:
                tiny_visuals.append({"page": page_id, "block_id": vb["block_id"], "area_ratio": round(ar, 4)})
            if ar > 0.65:
                huge_visuals.append({"page": page_id, "block_id": vb["block_id"], "area_ratio": round(ar, 4)})

        hs = cs = os = 0
        for tb in page.get("text_blocks", []):
            role = tb.get("role")
            txt = tb.get("text", "").strip()
            if role == "headline":
                hs += 1
                if re.match(r"^[-–—]", txt):
                    dialogue_headline.append({"page": page_id, "text": txt[:120]})
            elif role == "content":
                cs += 1
            else:
                os += 1
        headline_counts.append(hs)
        content_counts.append(cs)
        other_counts.append(os)

    return {
        "pages": len(pages),
        "visual": {
            "per_page": visual_counts,
            "total": int(sum(visual_counts)),
            "median_area_ratio": round(statistics.median(visual_area), 4) if visual_area else 0.0,
            "tiny_count(<0.03)": len(tiny_visuals),
            "huge_count(>0.65)": len(huge_visuals),
            "tiny_examples": tiny_visuals[:10],
            "huge_examples": huge_visuals[:10],
        },
        "text": {
            "headline_per_page": headline_counts,
            "content_per_page": content_counts,
            "other_per_page": other_counts,
            "dialogue_as_headline_count": len(dialogue_headline),
            "dialogue_as_headline_examples": dialogue_headline[:10],
        },
    }


def main() -> None:
    args = parse_args()
    pages = _load_pages(args.output)
    report = build_report(pages)
    out_path = args.report or (args.output / "pages" / "_benchmark" / "eval_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

