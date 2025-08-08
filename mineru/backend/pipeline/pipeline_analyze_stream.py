"""
Streaming pipeline for document analysis.

This module introduces a streaming variant of the doc analyze pipeline that
incrementally renders PDF pages and feeds them into the batch analyzer with
bounded queues and flush policies. It aims to:

- Reduce peak CPU memory usage by avoiding full materialization of all pages
- Reduce time-to-first-result by yielding page results as soon as available
- Preserve GPU efficiency by batching pages up to MINERU_MIN_BATCH_INFERENCE_SIZE

The original pipeline remains unchanged. This module can be integrated
incrementally without modifying existing call sites.

Resource constraints considered:
- CPU memory cap ~50 GiB (configurable via queue sizes and flush policies)
- GPU memory cap ~24 GiB (batch size is bounded by MINERU_MIN_BATCH_INFERENCE_SIZE)

Environment variables:
- MINERU_MIN_BATCH_INFERENCE_SIZE (int, default: 384)
- MINERU_STREAM_QUEUE_SIZE (int, default: 2048)
- MINERU_BATCH_FLUSH_MS (int, default: 500)

Usage pattern (example):
    for page_res in doc_analyze_stream(pdf_bytes_list, lang_list):
        # page_res: { 'pdf_idx', 'page_idx', 'layout_dets', 'page_info' }
        ...

If PDF-level aggregation is required, collect pages with the same pdf_idx and
assemble later using existing utilities like `pipeline_result_to_middle_json`.
"""

from __future__ import annotations

import os
import time
import threading
from queue import Queue, Empty
from typing import Any, Dict, Iterator, List, Tuple

from loguru import logger
from PIL import Image

from .pipeline_analyze import classify, batch_image_analyze
from mineru.utils.pdf_reader import load_images_from_pdf_iter


def doc_analyze_stream(
    pdf_bytes_list: List[bytes],
    lang_list: List[str],
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Stream pages through the batch analyzer and yield page-level results.

    - Producer incrementally renders PDF pages and enqueues them with metadata
    - Consumer aggregates into batches and invokes `batch_image_analyze`
    - Results are yielded per-page with original indices for optional reordering

    The function is designed to be backpressure-aware via a bounded queue to
    keep CPU memory usage under control. Batch size and flush interval are used
    to balance throughput and latency while respecting GPU memory constraints.
    """

    batch_size = int(os.environ.get("MINERU_MIN_BATCH_INFERENCE_SIZE", 384))
    queue_maxsize = int(os.environ.get("MINERU_STREAM_QUEUE_SIZE", 2048))
    flush_ms = int(os.environ.get("MINERU_BATCH_FLUSH_MS", 500))

    # Rough memory guidance:
    # Each PIL image at 144 DPI and 2560 long side may be ~3-15 MiB depending
    # on page content. queue_maxsize should be chosen to keep total under ~50 GiB.
    # E.g., 2048 * 8 MiB ~= 16 GiB upper bound in practice; adjust as needed.

    q: Queue[Tuple[int, int, Image.Image, bool, str] | None] = Queue(maxsize=queue_maxsize)
    stop_token: None = None

    def producer() -> None:
        for pdf_idx, pdf_bytes in enumerate(pdf_bytes_list):
            try:
                ocr_enable = False
                if parse_method == "auto":
                    ocr_enable = classify(pdf_bytes) == "ocr"
                elif parse_method == "ocr":
                    ocr_enable = True

                for page_idx, img_dict in enumerate(load_images_from_pdf_iter(pdf_bytes, dpi=200)):
                    pil_img = img_dict["img_pil"]
                    q.put((pdf_idx, page_idx, pil_img, ocr_enable, lang_list[pdf_idx]))
            except Exception as e:
                logger.warning(f"Skip PDF {pdf_idx} due to error: {e}")
                continue
        q.put(stop_token)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    batch_images: List[Tuple[Image.Image, bool, str]] = []
    batch_meta: List[Tuple[int, int, Tuple[int, int]]] = []  # (pdf_idx, page_idx, (w,h))
    last_flush = time.time()
    seen_stop = False

    def should_flush() -> str | None:
        if len(batch_images) >= batch_size:
            return "size"
        if batch_images and (time.time() - last_flush) * 1000 >= flush_ms:
            return "timeout"
        if seen_stop:
            return "drain"
        return None

    while True:
        try:
            item = q.get(timeout=flush_ms / 1000)
            if item is stop_token:
                seen_stop = True
            else:
                # mypy/pyright hint
                assert isinstance(item, tuple)
                pdf_idx, page_idx, pil_img, ocr_enable, lang = item
                batch_images.append((pil_img, ocr_enable, lang))
                batch_meta.append((pdf_idx, page_idx, (pil_img.width, pil_img.height)))
        except Empty:
            pass

        reason = should_flush()
        if reason:
            if batch_images:
                batch_start = time.time()
                results = batch_image_analyze(batch_images, formula_enable, table_enable)
                batch_dur = time.time() - batch_start
                logger.info(
                    f"Flushed batch: size={len(batch_images)}, reason={reason}, time={batch_dur:.2f}s"
                )
                for (pdf_idx, page_idx, (w, h)), layout_dets in zip(batch_meta, results):
                    yield {
                        "pdf_idx": pdf_idx,
                        "page_idx": page_idx,
                        "layout_dets": layout_dets,
                        "page_info": {"page_no": page_idx, "width": w, "height": h},
                    }
                batch_images.clear()
                batch_meta.clear()
            last_flush = time.time()

            if seen_stop and q.empty():
                break


