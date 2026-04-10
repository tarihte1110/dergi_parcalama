from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import load_config
from src.pipeline import DocumentPipeline
from src.utils.logging_utils import setup_logger
from src.utils.pdf_ops import iter_pdf_pages_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-page Document AI pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input image or pdf path")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path")
    parser.add_argument("--page", type=int, default=None, help="If input is PDF, process only this 1-based page")
    parser.add_argument("--debug", action="store_true", help="Force debug outputs")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug outputs")
    parser.add_argument("--show-progress", action="store_true", help="Show third-party progress bars")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.runtime.suppress_progress_bars and not args.show_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")
    logger = setup_logger(log_file=args.output / "run.log")
    pipeline = DocumentPipeline(config=config, logger=logger)

    debug_override = None
    if args.debug:
        debug_override = True
    elif args.no_debug:
        debug_override = False

    if args.input.suffix.lower() == ".pdf":
        pages = iter_pdf_pages_bgr(args.input, scale=2.0)
        selected = pages
        if args.page is not None:
            selected = [p for p in pages if p[0] == args.page]
            if not selected:
                raise ValueError(f"Requested page {args.page} not found in {args.input}")
        for page_idx, page_bgr in selected:
            stem = f"{args.input.stem}_p{page_idx:04d}"
            source_ref = f"{args.input.as_posix()}#page={page_idx}"
            payload = pipeline.process_image_array(
                image=page_bgr,
                output_root=args.output,
                source_ref=source_ref,
                stem=stem,
                debug_override=debug_override,
            )
            logger.info(
                "Processed %s#page=%d | text_blocks=%d visual_blocks=%d",
                args.input,
                page_idx,
                len(payload.get("text_blocks", [])),
                len(payload.get("visual_blocks", [])),
            )
    else:
        payload = pipeline.process_image(image_path=args.input, output_root=args.output, debug_override=debug_override)
        logger.info(
            "Processed %s | text_blocks=%d visual_blocks=%d",
            args.input,
            len(payload.get("text_blocks", [])),
            len(payload.get("visual_blocks", [])),
        )


if __name__ == "__main__":
    main()
