from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.pipeline import DocumentPipeline
from src.utils.io import SUPPORTED_IMAGE_EXTS, SUPPORTED_PDF_EXTS, list_input_images, list_input_pdfs
from src.utils.logging_utils import setup_logger
from src.utils.pdf_ops import iter_pdf_pages_bgr_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Document AI pipeline")
    parser.add_argument("--input", type=Path, default=Path("images"), help="Input image directory")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path")
    parser.add_argument("--debug", action="store_true", help="Force debug outputs")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug outputs")
    parser.add_argument(
        "--ocr-page-batch",
        type=int,
        default=None,
        help="Optional PDF page micro-batch size for OCR throughput (defaults to config.ocr.surya_page_batch_size).",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show third-party progress bars")
    return parser.parse_args()


def _process_pdf_chunk(
    pipeline: DocumentPipeline,
    pdf_path: Path,
    output_root: Path,
    debug_override: bool | None,
    chunk: list[tuple[int, object]],
    logger,
) -> tuple[int, int]:
    if not chunk:
        return 0, 0

    ok_count = 0
    fail_count = 0
    images = [img for _, img in chunk]
    try:
        batch_lines = pipeline.backend.detect_batch(images)
        if len(batch_lines) != len(images):
            logger.warning("OCR batch output size mismatch for %s, falling back to per-page OCR.", pdf_path)
            batch_lines = [None for _ in images]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("OCR batch failed for %s, fallback per-page OCR: %s", pdf_path, exc)
        batch_lines = [None for _ in images]

    for (page_idx, page_bgr), ocr_lines in zip(chunk, batch_lines):
        try:
            stem = f"{pdf_path.stem}_p{page_idx:04d}"
            source_ref = f"{pdf_path.as_posix()}#page={page_idx}"
            pipeline.process_image_array(
                image=page_bgr,
                output_root=output_root,
                source_ref=source_ref,
                stem=stem,
                debug_override=debug_override,
                ocr_lines=ocr_lines,
            )
            ok_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            fail_count += 1
            logger.exception("Failed to process %s#page=%d: %s", pdf_path, page_idx, exc)

    return ok_count, fail_count


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    quiet_progress = config.runtime.suppress_progress_bars and not args.show_progress
    if quiet_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")
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

    for image_path in tqdm(images, desc="Processing image pages", disable=quiet_progress):
        try:
            pipeline.process_image(image_path=image_path, output_root=args.output, debug_override=debug_override)
            ok_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            fail_count += 1
            logger.exception("Failed to process %s: %s", image_path, exc)

    for pdf_path in tqdm(pdfs, desc="Processing PDFs", disable=quiet_progress):
        try:
            page_batch = int(args.ocr_page_batch or config.ocr.surya_page_batch_size)
            page_batch = max(1, page_batch)
            chunk: list[tuple[int, object]] = []
            for page_idx, page_bgr in iter_pdf_pages_bgr_stream(pdf_path, scale=2.0):
                chunk.append((page_idx, page_bgr))
                if len(chunk) >= page_batch:
                    ok, fail = _process_pdf_chunk(
                        pipeline=pipeline,
                        pdf_path=pdf_path,
                        output_root=args.output,
                        debug_override=debug_override,
                        chunk=chunk,
                        logger=logger,
                    )
                    ok_count += ok
                    fail_count += fail
                    chunk = []
            if chunk:
                ok, fail = _process_pdf_chunk(
                    pipeline=pipeline,
                    pdf_path=pdf_path,
                    output_root=args.output,
                    debug_override=debug_override,
                    chunk=chunk,
                    logger=logger,
                )
                ok_count += ok
                fail_count += fail
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to process %s: %s", pdf_path, exc)
            fail_count += 1

    logger.info("Batch finished. success=%d failed=%d total=%d", ok_count, fail_count, ok_count + fail_count)


if __name__ == "__main__":
    main()
