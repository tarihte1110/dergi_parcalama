from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}


def list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def list_input_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_PDF_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "pages": output_root / "pages",
        "images": output_root / "images",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
