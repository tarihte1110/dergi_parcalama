from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any
import warnings

import cv2
from tqdm import tqdm

from src.config import load_config
from src.pipeline import DocumentPipeline
from src.utils.io import SUPPORTED_IMAGE_EXTS, SUPPORTED_PDF_EXTS, list_input_images, list_input_pdfs
from src.utils.logging_utils import setup_logger
from src.utils.pdf_ops import iter_pdf_pages_bgr_stream


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_run_lock(lock_path: Path) -> int | None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{current_pid}\n".encode("utf-8"))
            os.fsync(fd)
            return fd
        except FileExistsError:
            existing_pid = -1
            try:
                raw = lock_path.read_text(encoding="utf-8").strip()
                existing_pid = int(raw.splitlines()[0]) if raw else -1
            except Exception:
                existing_pid = -1

            if existing_pid > 0 and _is_pid_alive(existing_pid):
                return None

            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return None


def _release_run_lock(lock_fd: int | None, lock_path: Path) -> None:
    if lock_fd is None:
        return
    try:
        os.close(lock_fd)
    except OSError:
        pass

    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
        owner_pid = int(raw.splitlines()[0]) if raw else -1
    except Exception:
        owner_pid = -1

    if owner_pid == os.getpid():
        try:
            lock_path.unlink()
        except OSError:
            pass


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
    parser.add_argument(
        "--pdf-scale",
        type=float,
        default=1.3,
        help="PDF render scale (lower = lower memory / faster, higher = better OCR detail).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU thread count for OpenCV/Torch CPU ops (0=auto/os.cpu_count).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Process at most this many pages/images from input ordering (0=all).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="For PDF input, start from this 1-based page index.",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show third-party progress bars")
    return parser.parse_args()


def _cleanup_memory(pipeline: DocumentPipeline) -> None:
    gc.collect()
    torch_mod = getattr(pipeline.backend, "_torch", None)
    if torch_mod is not None and hasattr(torch_mod, "mps"):
        try:
            torch_mod.mps.empty_cache()
        except Exception:  # pragma: no cover
            pass


def _write_page_profile(output_root: Path, profile: dict[str, Any]) -> None:
    page_dir = output_root / "pages" / "_benchmark" / "page_profiles"
    page_dir.mkdir(parents=True, exist_ok=True)
    stem = str(profile.get("stem", "page"))
    path = page_dir / f"{stem}.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _configure_runtime_threads(pipeline: DocumentPipeline, cpu_threads_arg: int, logger) -> int:
    cpu_threads = int(cpu_threads_arg) if cpu_threads_arg and cpu_threads_arg > 0 else int(os.cpu_count() or 1)
    cpu_threads = max(1, cpu_threads)

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        cv2.setNumThreads(cpu_threads)
    except Exception:  # pragma: no cover
        pass

    torch_mod = getattr(pipeline.backend, "_torch", None)
    if torch_mod is not None:
        try:
            torch_mod.set_num_threads(cpu_threads)
        except Exception:  # pragma: no cover
            pass
        try:
            torch_mod.set_num_interop_threads(min(4, cpu_threads))
        except Exception:  # pragma: no cover
            pass

    logger.info("Runtime threads configured: cpu_threads=%d", cpu_threads)
    return cpu_threads


def _process_pdf_chunk(
    pipeline: DocumentPipeline,
    pdf_path: Path,
    output_root: Path,
    debug_override: bool | None,
    chunk: list[tuple[int, object]],
    logger,
) -> tuple[int, int, list[dict[str, Any]]]:
    if not chunk:
        return 0, 0, []

    ok_count = 0
    fail_count = 0
    page_profiles: list[dict[str, Any]] = []
    images = [img for _, img in chunk]
    ocr_elapsed = 0.0
    try:
        t_ocr0 = time.perf_counter()
        batch_lines = pipeline.backend.detect_batch(images)
        ocr_elapsed = time.perf_counter() - t_ocr0
        if len(batch_lines) != len(images):
            logger.warning("OCR batch output size mismatch for %s, falling back to per-page OCR.", pdf_path)
            batch_lines = [None for _ in images]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("OCR batch failed for %s, fallback per-page OCR: %s", pdf_path, exc)
        batch_lines = [None for _ in images]

    shared_ocr_sec = (ocr_elapsed / len(images)) if (images and ocr_elapsed > 0.0) else 0.0
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
                external_ocr_sec=shared_ocr_sec if ocr_lines is not None else 0.0,
            )
            page_profile = dict(pipeline.last_page_profile) if pipeline.last_page_profile else {}
            if page_profile:
                page_profile["input_kind"] = "pdf"
                page_profiles.append(page_profile)
                _write_page_profile(output_root, page_profile)
                logger.info(
                    "Page done %s | total=%.2fs ocr=%.2fs text=%.2fs visual=%.2fs",
                    page_profile.get("stem", stem),
                    float(page_profile.get("total_sec", 0.0)),
                    float(page_profile.get("ocr_sec", 0.0)),
                    float(page_profile.get("text_group_classify_sec", 0.0)),
                    float(page_profile.get("visual_extract_sec", 0.0)),
                )
            ok_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            fail_count += 1
            logger.exception("Failed to process %s#page=%d: %s", pdf_path, page_idx, exc)
        finally:
            del page_bgr

    _cleanup_memory(pipeline)
    return ok_count, fail_count, page_profiles


def _build_benchmark_summary(page_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not page_profiles:
        return {
            "pages": 0,
            "total_sec": 0.0,
            "avg_sec_per_page": 0.0,
            "median_sec_per_page": 0.0,
            "p95_sec_per_page": 0.0,
            "throughput_pages_per_min": 0.0,
            "stage_totals_sec": {},
        }

    totals = [float(p.get("total_sec", 0.0)) for p in page_profiles]
    sorted_totals = sorted(totals)
    p95_idx = min(len(sorted_totals) - 1, max(0, int(round(0.95 * (len(sorted_totals) - 1)))))

    stage_keys = [
        "read_sec",
        "ocr_sec",
        "prep_sec",
        "text_group_classify_sec",
        "mask_sec",
        "visual_extract_sec",
        "crop_save_sec",
        "json_write_sec",
        "debug_sec",
    ]
    stage_totals = {
        key: round(sum(float(p.get(key, 0.0)) for p in page_profiles), 4)
        for key in stage_keys
    }

    total_sec = sum(totals)
    pages = len(page_profiles)
    return {
        "pages": pages,
        "total_sec": round(total_sec, 4),
        "avg_sec_per_page": round(total_sec / pages, 4),
        "median_sec_per_page": round(float(statistics.median(totals)), 4),
        "p95_sec_per_page": round(float(sorted_totals[p95_idx]), 4),
        "throughput_pages_per_min": round((pages * 60.0) / max(total_sec, 1e-9), 3),
        "stage_totals_sec": stage_totals,
    }


def _write_benchmark_report(output_root: Path, page_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _build_benchmark_summary(page_profiles)
    report = {"summary": summary, "pages": page_profiles}
    report_root = output_root / "pages" / "_benchmark"
    report_root.mkdir(parents=True, exist_ok=True)

    report_json = report_root / "benchmark_report.json"
    report_md = report_root / "benchmark_report.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Benchmark Report",
        "",
        f"- Pages: {summary['pages']}",
        f"- Total: {summary['total_sec']} sec",
        f"- Avg/page: {summary['avg_sec_per_page']} sec",
        f"- Median/page: {summary['median_sec_per_page']} sec",
        f"- P95/page: {summary['p95_sec_per_page']} sec",
        f"- Throughput: {summary['throughput_pages_per_min']} pages/min",
        "",
        "## Stage Totals (sec)",
    ]
    for key, value in summary["stage_totals_sec"].items():
        lines.append(f"- {key}: {value}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module=r"multiprocessing\.resource_tracker")
    config = load_config(args.config)
    quiet_progress = config.runtime.suppress_progress_bars and not args.show_progress
    if quiet_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")
    logger = setup_logger(log_file=args.output / "run.log")
    lock_path = Path.cwd() / ".document_ai_run.lock"
    lock_fd = _acquire_run_lock(lock_path)
    if lock_fd is None:
        logger.error("Another run is already active. lock=%s", lock_path)
        return

    try:
        pipeline = DocumentPipeline(config=config, logger=logger)
        _configure_runtime_threads(pipeline, args.cpu_threads, logger)

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

        max_pages = max(0, int(args.max_pages))
        page_budget = max_pages if max_pages > 0 else None
        seen_pages = 0

        ok_count = 0
        fail_count = 0
        page_profiles: list[dict[str, Any]] = []

        for image_path in tqdm(images, desc="Processing image pages", disable=quiet_progress):
            if page_budget is not None and seen_pages >= page_budget:
                break
            seen_pages += 1
            try:
                pipeline.process_image(image_path=image_path, output_root=args.output, debug_override=debug_override)
                profile = dict(pipeline.last_page_profile) if pipeline.last_page_profile else {}
                if profile:
                    profile["input_kind"] = "image"
                    page_profiles.append(profile)
                    _write_page_profile(args.output, profile)
                    logger.info(
                        "Page done %s | total=%.2fs ocr=%.2fs text=%.2fs visual=%.2fs",
                        profile.get("stem", image_path.stem),
                        float(profile.get("total_sec", 0.0)),
                        float(profile.get("ocr_sec", 0.0)),
                        float(profile.get("text_group_classify_sec", 0.0)),
                        float(profile.get("visual_extract_sec", 0.0)),
                    )
                    _write_benchmark_report(args.output, page_profiles)
                ok_count += 1
            except Exception as exc:  # pylint: disable=broad-except
                fail_count += 1
                logger.exception("Failed to process %s: %s", image_path, exc)

        for pdf_path in tqdm(pdfs, desc="Processing PDFs", disable=quiet_progress):
            if page_budget is not None and seen_pages >= page_budget:
                break
            try:
                if args.ocr_page_batch is not None:
                    page_batch = int(args.ocr_page_batch)
                else:
                    backend_name = config.ocr.backend.lower().strip()
                    default_batch = config.ocr.surya_page_batch_size if backend_name == "surya" else 1
                    # Keep default conservative on MPS to avoid OOM spikes.
                    page_batch = 1 if backend_name == "surya" else default_batch
                page_batch = max(1, page_batch)
                chunk: list[tuple[int, Any]] = []
                start_page = max(1, int(args.start_page))
                for page_idx, page_bgr in iter_pdf_pages_bgr_stream(pdf_path, scale=float(args.pdf_scale)):
                    if page_idx < start_page:
                        continue
                    if page_budget is not None and (seen_pages + len(chunk)) >= page_budget:
                        break
                    chunk.append((page_idx, page_bgr))
                    if len(chunk) >= page_batch:
                        seen_pages += len(chunk)
                        ok, fail, profiles = _process_pdf_chunk(
                            pipeline=pipeline,
                            pdf_path=pdf_path,
                            output_root=args.output,
                            debug_override=debug_override,
                            chunk=chunk,
                            logger=logger,
                        )
                        ok_count += ok
                        fail_count += fail
                        page_profiles.extend(profiles)
                        _write_benchmark_report(args.output, page_profiles)
                        chunk = []
                if chunk:
                    seen_pages += len(chunk)
                    ok, fail, profiles = _process_pdf_chunk(
                        pipeline=pipeline,
                        pdf_path=pdf_path,
                        output_root=args.output,
                        debug_override=debug_override,
                        chunk=chunk,
                        logger=logger,
                    )
                    ok_count += ok
                    fail_count += fail
                    page_profiles.extend(profiles)
                    _write_benchmark_report(args.output, page_profiles)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to process %s: %s", pdf_path, exc)
                fail_count += 1

        report = _write_benchmark_report(args.output, page_profiles)
        summary = report["summary"]
        logger.info("Batch finished. success=%d failed=%d total=%d", ok_count, fail_count, ok_count + fail_count)
        logger.info(
            "Benchmark: pages=%d total=%.2fs avg=%.2fs/page p95=%.2fs throughput=%.2f page/min",
            summary["pages"],
            summary["total_sec"],
            summary["avg_sec_per_page"],
            summary["p95_sec_per_page"],
            summary["throughput_pages_per_min"],
        )
    finally:
        _release_run_lock(lock_fd, lock_path)


if __name__ == "__main__":
    main()
