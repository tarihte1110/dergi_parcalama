from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.pipeline import DocumentPipeline
from src.utils.io import SUPPORTED_IMAGE_EXTS, SUPPORTED_PDF_EXTS, list_input_images, list_input_pdfs
from src.utils.logging_utils import setup_logger
from src.utils.pdf_ops import iter_pdf_pages_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Document AI pipeline")
    parser.add_argument("--input", type=Path, default=Path("images"), help="Input image directory")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path")
    parser.add_argument("--debug", action="store_true", help="Force debug outputs")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(log_file=args.output / "run.log")

    pipeline = DocumentPipeline(config=config, logger=logger)

    debug_override = None
    if args.debug:
        debug_override = True
    elif args.no_debug:
        debug_override = False

    images: list[Path] = []
    pdfs: list[Path] = []
    if args.input.is_file():
        ext = args.input.suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            images = [args.input]
        elif ext in SUPPORTED_PDF_EXTS:
            pdfs = [args.input]
    else:
        images = list_input_images(args.input)
        pdfs = list_input_pdfs(args.input)

    if not images and not pdfs:
        logger.warning("No image/pdf files found in %s", args.input)
        return

    ok_count = 0
    fail_count = 0

    for image_path in tqdm(images, desc="Processing image pages"):
        try:
            pipeline.process_image(image_path=image_path, output_root=args.output, debug_override=debug_override)
            ok_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            fail_count += 1
            logger.exception("Failed to process %s: %s", image_path, exc)

    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            pages = iter_pdf_pages_bgr(pdf_path, scale=2.0)
            for page_idx, page_bgr in pages:
                stem = f"{pdf_path.stem}_p{page_idx:04d}"
                source_ref = f"{pdf_path.as_posix()}#page={page_idx}"
                pipeline.process_image_array(
                    image=page_bgr,
                    output_root=args.output,
                    source_ref=source_ref,
                    stem=stem,
                    debug_override=debug_override,
                )
                ok_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            fail_count += 1
            logger.exception("Failed to process %s: %s", pdf_path, exc)

    logger.info("Batch finished. success=%d failed=%d total=%d", ok_count, fail_count, ok_count + fail_count)


if __name__ == "__main__":
    main()
