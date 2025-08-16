import os
import time
import threading
from typing import List, Tuple, Optional, Dict, Any, Iterable
import PIL.Image
from mineru.utils.logger_utils import get_logger
logger = get_logger("mineru.stream_pipeline")

from .model_init import MineruPipelineModel
from mineru.utils.config_reader import get_device
from ...utils.pdf_classify import classify
from ...utils.pdf_image_tools import load_images_from_pdf
from ...utils.model_utils import get_vram, clean_memory

class ModelSingleton:
    _instance = None
    _models = {}
    _tls = threading.local()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_current_device(self, device: Optional[str]) -> None:
        self._tls.current_device = device

    def get_model(
        self,
        lang=None,
        formula_enable: bool = True,
        table_enable: bool = True,
        device_override: Optional[str] = None,
    ):
        device = device_override or getattr(self._tls, 'current_device', None) or get_device()
        key = (device, lang, formula_enable, table_enable)
        if key not in self._models:
            self._models[key] = custom_model_init(
                lang=lang,
                formula_enable=formula_enable,
                table_enable=table_enable,
                device_override=device,
            )
        return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=True,
    table_enable=True,
    device_override: Optional[str] = None,
):
    model_init_start = time.time()
    # 设备选择（允许覆盖）
    device = device_override or get_device()

    formula_config = {"enable": formula_enable}
    table_config = {"enable": table_enable}

    model_input = {
        'device': device,
        'table_config': table_config,
        'formula_config': formula_config,
        'lang': lang,
    }

    custom_model = MineruPipelineModel(**model_input)

    model_init_cost = time.time() - model_init_start
    logger.info(f'model init cost: {model_init_cost}')

    return custom_model
 


def batch_image_analyze(
        images_with_extra_info: List[Tuple[PIL.Image.Image, bool, str]],
        formula_enable=True,
        table_enable=True,
        device_override: Optional[str] = None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)

    from .batch_analyze import BatchAnalyze

    model_manager = ModelSingleton()

    batch_ratio = 1
    device = device_override or get_device()

    if str(device).startswith('npu'):
        try:
            import torch_npu  # type: ignore
            if torch_npu.npu.is_available():
                torch_npu.npu.set_compile_mode(jit_compile=False)
        except Exception as e:
            raise RuntimeError(
                "NPU is selected as device, but torch_npu is not available. "
                "Please ensure that the torch_npu package is installed correctly."
            ) from e

    if str(device).startswith('npu') or str(device).startswith('cuda'):
        vram = get_vram(device)
        if vram is not None:
            gpu_memory = int(os.getenv('MINERU_VIRTUAL_VRAM_SIZE', round(vram)))
            if gpu_memory >= 16:
                batch_ratio = 16
            elif gpu_memory >= 12:
                batch_ratio = 8
            elif gpu_memory >= 8:
                batch_ratio = 4
            elif gpu_memory >= 6:
                batch_ratio = 2
            else:
                batch_ratio = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')
        else:
            # Default batch_ratio when VRAM can't be determined
            batch_ratio = 1
            logger.info(f'Could not determine GPU memory, using default batch_ratio: {batch_ratio}')

    batch_model = BatchAnalyze(model_manager, batch_ratio, formula_enable, table_enable)
    #76%
    results = batch_model(images_with_extra_info)

    clean_memory(device)

    return results


# =============================
# Streaming pipeline (best practice skeleton)
# =============================

class _PageTask:
    """A minimal unit flowing through the streaming pipeline.
    Holds CPU image (PIL) and optional GPU payloads for zero-copy pass-through between stages.
    """

    def __init__(self, pdf_idx: int, page_idx: int, pil_img: PIL.Image.Image, lang: str, ocr_enable: bool) -> None:
        self.pdf_idx = pdf_idx
        self.page_idx = page_idx
        self.pil_img = pil_img
        self.lang = lang
        self.ocr_enable = ocr_enable
        # Stage-wise artifacts; stages may store GPU tensors here to avoid D2H/H2D
        self.gpu_payload: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {"width": pil_img.width, "height": pil_img.height}


class _MicroBatch:
    def __init__(self, tasks: List[_PageTask]):
        self.tasks = tasks


def stream_doc_analyze(
    pdf_bytes_iterable: Iterable[bytes],
    lang_iterable: Iterable[str],
    parse_method: str = 'auto',
    formula_enable: bool = True,
    table_enable: bool = True,
    micro_batch_size: int = 64,
    devices_plan: Optional[Dict[str, Any]] = None,
    prefetch_micro_batches: int = 4,
    enable_gpu_pass_through: bool = False,
):
    """
    Streaming doc_analyze generator.

    Current working implementation: two-stage streaming
      - Stage-GPU: batch_image_analyze (existing black-box multi-task GPU inference)
      - Stage-CPU: result construction

    This skeleton is designed to be extended to true multi-stage pipeline (layout/MFD/MFR/OCR-det/OCR-rec)
    when stage-specific APIs are exposed. For now, it achieves:
      - Micro-batching
      - Multi-GPU data-parallel across micro-batches (via devices_plan['analyze_replicas'])
      - PDF streaming input; results yielded per-PDF when all its pages are done
    """
    import threading
    import queue

    # Default plan: replicate analyze stage across visible CUDA devices (or single device)
    default_plan = {
        "analyze_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if os.environ.get("CUDA_VISIBLE_DEVICES") else ["0"],
        "analyze_replicas": None,  # if None -> len(analyze_devices)
    }
    plan = devices_plan or default_plan
    analyze_devices = [d.strip() for d in plan.get("analyze_devices", default_plan["analyze_devices"]) if d.strip() != ""]
    if not analyze_devices:
        analyze_devices = ["0"]
    analyze_replicas = plan.get("analyze_replicas") or len(analyze_devices)

    # Queues (Optional[...] to allow None as stop token)
    q_pages: "queue.Queue[Optional[_PageTask]]" = queue.Queue(maxsize=prefetch_micro_batches * micro_batch_size)
    q_micro_batches: "queue.Queue[Optional[_MicroBatch]]" = queue.Queue(maxsize=prefetch_micro_batches)
    q_results: "queue.Queue[Tuple[int, Dict[str, Any]]]" = queue.Queue()

    stop_token: Optional[_MicroBatch] = None

    pdf_bytes_list = list(pdf_bytes_iterable)
    lang_list = list(lang_iterable)

    # Preload PDF pages and docs to avoid race and double loading
    preloaded_images: Dict[int, Any] = {}
    preloaded_docs: Dict[int, Any] = {}
    ocr_enabled_list: Dict[int, bool] = {}
    lang_map: Dict[int, str] = {}
    for pdf_idx, pdf_bytes in enumerate(pdf_bytes_list):
        _ocr_enable = False
        if parse_method == 'auto':
            if classify(pdf_bytes) == 'ocr':
                _ocr_enable = True
        elif parse_method == 'ocr':
            _ocr_enable = True
        _lang = lang_list[pdf_idx]
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes)
        preloaded_images[pdf_idx] = images_list
        preloaded_docs[pdf_idx] = pdf_doc
        ocr_enabled_list[pdf_idx] = _ocr_enable
        lang_map[pdf_idx] = _lang

    # Build PDF docs and pages (producer) from preloaded data
    def producer() -> None:
        for pdf_idx in range(len(pdf_bytes_list)):
            images_list = preloaded_images[pdf_idx]
            _lang = lang_map[pdf_idx]
            _ocr_enable = ocr_enabled_list[pdf_idx]
            for page_idx in range(len(images_list)):
                img_dict = images_list[page_idx]
                task = _PageTask(pdf_idx, page_idx, img_dict['img_pil'], _lang, _ocr_enable)
                q_pages.put(task)
        # Signal end
        for _ in range(analyze_replicas):
            q_pages.put(stop_token)

    # Micro-batcher (decouples producer and GPU workers)
    def micro_batcher() -> None:
        current: List[_PageTask] = []
        finished_replicas = 0
        while True:
            item = q_pages.get()
            if item is stop_token:
                finished_replicas += 1
                if current:
                    q_micro_batches.put(_MicroBatch(current))
                    current = []
                if finished_replicas >= analyze_replicas:
                    q_micro_batches.put(stop_token)
                    break
                else:
                    # There may be more stop tokens for other replicas; continue
                    continue
            assert isinstance(item, _PageTask)
            current.append(item)
            if len(current) >= micro_batch_size:
                q_micro_batches.put(_MicroBatch(current))
                current = []

    # GPU analyze workers (data parallel across GPUs)
    def gpu_worker(worker_idx: int, device_id: str) -> None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.set_device(int(device_id))
        except Exception:
            pass
        # Prepare per-worker model via existing singleton
        model_manager = ModelSingleton()
        # Bind thread-local device & warmup model instance on this device
        device_str = f"cuda:{int(device_id)}" if device_id.isdigit() else device_id
        model_manager.set_current_device(device_str)
        model_manager.get_model(lang=None, formula_enable=formula_enable, table_enable=table_enable, device_override=device_str)

        while True:
            mb = q_micro_batches.get()
            if mb is stop_token:
                q_micro_batches.put(stop_token)  # propagate for other workers
                break
            assert isinstance(mb, _MicroBatch)
            images_with_extra_info = [
                (t.pil_img, t.ocr_enable, t.lang) for t in mb.tasks
            ]
            batch_results = batch_image_analyze(
                images_with_extra_info,
                formula_enable=formula_enable,
                table_enable=table_enable,
                device_override=device_str,
            )
            for t, r in zip(mb.tasks, batch_results):
                # Emit page-level result immediately for assembly
                q_results.put((t.pdf_idx, {
                    "page_no": t.page_idx,
                    "page_info": {"width": t.meta["width"], "height": t.meta["height"]},
                    "layout_dets": r,
                }))

    # Result aggregator to yield per-PDF results when complete
    def aggregator() -> Iterable[Tuple[int, Dict[str, Any]]]:
        # Use preloaded info
        page_counts: Dict[int, int] = {i: len(preloaded_images[i]) for i in range(len(pdf_bytes_list))}

        collected: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(pdf_bytes_list))}
        finished_pages: Dict[int, int] = {i: 0 for i in range(len(pdf_bytes_list))}
        total_pages = sum(page_counts.values())
        emitted = set()
        seen = 0
        while seen < total_pages:
            pdf_idx, page_dict = q_results.get()
            seen += 1
            collected[pdf_idx].append(page_dict)
            finished_pages[pdf_idx] += 1
            if finished_pages[pdf_idx] >= page_counts[pdf_idx] and pdf_idx not in emitted:
                emitted.add(pdf_idx)
                # Sort by page_no
                collected[pdf_idx].sort(key=lambda x: x["page_no"])
                yield pdf_idx, {
                    "pages": collected[pdf_idx],
                    "images_list": preloaded_images[pdf_idx],
                    "pdf_doc": preloaded_docs[pdf_idx],
                    "lang": lang_map[pdf_idx],
                    "ocr_enable": ocr_enabled_list[pdf_idx],
                }

    # Threads
    th_producer = threading.Thread(target=producer, daemon=True)
    th_batcher = threading.Thread(target=micro_batcher, daemon=True)
    th_workers = [
        threading.Thread(target=gpu_worker, args=(i, analyze_devices[i % len(analyze_devices)]), daemon=True)
        for i in range(analyze_replicas)
    ]

    th_producer.start()
    th_batcher.start()
    for th in th_workers:
        th.start()

    # Now stream out results per PDF
    for pdf_idx, result in aggregator():
        yield pdf_idx, result

    # Join threads (best-effort; they are daemons)
    try:
        th_producer.join(timeout=0.1)
        th_batcher.join(timeout=0.1)
        for th in th_workers:
            th.join(timeout=0.1)
    except Exception:
        pass
