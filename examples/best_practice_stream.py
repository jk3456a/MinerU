import copy
import glob
import os
import random
import traceback

os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")
os.environ.setdefault("MINERU_VIRTUAL_VRAM_SIZE", "24")
os.environ.setdefault("MINERU_LOG_ENABLE", "1")
os.environ.setdefault("MINERU_LAYOUT_BATCH_SIZE", "16")
os.environ.setdefault("MINERU_MFD_BATCH_SIZE", "2")
os.environ.setdefault("MINERU_MFR_BATCH_SIZE", "32")
os.environ.setdefault("MINERU_OCR_DET_BATCH_SIZE", "32")
os.environ.setdefault("MINERU_STREAM_MICRO_BATCH_SIZE", "128")
import json

import time
import uuid
from typing import Any, cast
from mineru.utils.logger_utils import get_logger, set_run_id

from mineru.backend.pipeline.model_init import MineruPipelineModel
from mineru.data.data_reader_writer import FileBasedDataWriter

from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.stream_pipeline import stream_doc_analyze
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
import mineru.utils.nvtx_utils as nvtxu

logger = get_logger("mineru.examples.best_practice_stream")

_LOG_ENABLE = os.environ.get("MINERU_LOG_ENABLE", "1") == "1"

# 生成运行标识符
run_id = str(uuid.uuid4())[:8]  # 取前8位作为运行ID

if _LOG_ENABLE:
    # logger_utils 已配置轮转与控制台输出
    os.makedirs("logs", exist_ok=True)

# 运行开始标识（简单）
set_run_id(run_id)
logger.info(f"RUN START - {run_id}")


def infer_one_pdf(pdf_file_path, lang="ch"):
    with open(pdf_file_path, 'rb') as fi:
        pdf_bytes = fi.read()
    new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
    
    formula_enable = True
    table_enable = False
    pdf_name = os.path.basename(pdf_file_path)
    
    logger.info(f"Processing PDF: {pdf_name}")
    logger.info(f"Formula enable: {formula_enable}, Table enable: {table_enable}")
    
    t2_1 = time.time()
    with nvtxu.nvtx_range("pipeline_doc_analyze: " + pdf_name):
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                [new_pdf_bytes],
                [lang],
                parse_method="ocr",
                formula_enable=formula_enable, table_enable=table_enable
            )
        )
    t3 = time.time()
    
    # 计算页数
    total_pages = len(all_image_lists[0]) if all_image_lists else 0
    logger.info(f"Total pages: {total_pages}, time: {t3 - t2_1:.2f}s, throughput: {total_pages/(t3 - t2_1):.1f} pages/s" if total_pages > 0 else "No pages")
    
    model_list = infer_results[0]
    images_list = all_image_lists[0]
    pdf_doc = all_pdf_docs[0]
    _ocr_enable = True

    model_json = copy.deepcopy(model_list)

    local_image_dir = f"/cache/lizhen/repos/mineru/MinerU/output/best_practice/{pdf_name}"
    if not os.path.exists(local_image_dir):
        os.system(f"mkdir -p {local_image_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)

    t3_1 = time.time()
    with nvtxu.nvtx_range("processing.overall"):
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            lang, _ocr_enable, formula_enable
        )
    t4 = time.time()
    logger.info(f"Pipeline result to middle json spend: {t4 - t3_1:.2f}s")
    
    total_time = t4 - t0
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Average time per page (total): {total_time/total_pages:.3f}s" if total_pages > 0 else "No pages")
    logger.info(f"Overall throughput: {total_pages/total_time:.1f} pages/s" if total_pages > 0 else "No pages")
    
    ocr_result = {
        "middle_json": middle_json,
        "model_json": model_json
    }

    return ocr_result


def infer_many_pdfs_stream(pdf_file_paths, save_dir, lang="ch"):
    t0 = time.time()
    pdf_names = [os.path.basename(p) for p in pdf_file_paths]
    pdf_bytes_list = []
    for p in pdf_file_paths:
        with open(p, 'rb') as fi:
            pdf_bytes = fi.read()
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
        pdf_bytes_list.append(new_pdf_bytes)

    lang_list = [lang for _ in pdf_bytes_list]
    micro_batch_size = int(os.environ.get("MINERU_STREAM_MICRO_BATCH_SIZE", "64"))
    prefetch_mb = int(os.environ.get("MINERU_STREAM_PREFETCH", "4"))
    analyze_devices_env = os.environ.get("MINERU_STREAM_ANALYZE_DEVICES")
    devices_plan = None
    if analyze_devices_env:
        devices_plan = {"analyze_devices": analyze_devices_env.split(",")}

    # 过滤掉读取失败的
    valid_pairs = [(b, l, i) for i, (b, l) in enumerate(zip(pdf_bytes_list, lang_list)) if b is not None]
    if not valid_pairs:
        logger.warning("No valid PDFs to process in streaming mode.")
        return

    ordered_results = {}
    total_pages = 0
    t1 = time.time()
    for pdf_idx, result in stream_doc_analyze(
        (b for b, _, _ in valid_pairs),
        (l for _, l, _ in valid_pairs),
        parse_method="ocr",
        formula_enable=True,
        table_enable=False,
        micro_batch_size=micro_batch_size,
        devices_plan=devices_plan,
        prefetch_micro_batches=prefetch_mb,
        enable_gpu_pass_through=False,
    ):
        # 组装与保存（对齐串行API输出）
        pages = result["pages"]
        images_list = result["images_list"]
        pdf_doc = result["pdf_doc"]
        _lang = result["lang"]
        _ocr_enable = result["ocr_enable"]
        model_list = pages
        model_json = copy.deepcopy(model_list)

        pdf_name = pdf_names[valid_pairs[pdf_idx][2]]
        local_image_dir = f"/cache/lizhen/repos/mineru/MinerU/output/best_practice/{pdf_name}"
        if not os.path.exists(local_image_dir):
            os.system(f"mkdir -p {local_image_dir}")
        image_writer = FileBasedDataWriter(local_image_dir)

        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
        )

        ocr_result = {
            "middle_json": middle_json,
            "model_json": model_json,
            "pdf_path": pdf_file_paths[valid_pairs[pdf_idx][2]],
        }

        target_file = f"{save_dir}/{pdf_name}.json"
        os.makedirs(save_dir, exist_ok=True)
        with open(target_file, "w") as fo:
            fo.write(json.dumps(ocr_result, ensure_ascii=False))

        total_pages += len(images_list)
        ordered_results[pdf_idx] = target_file
        logger.info(f"[stream] Saved: {target_file}")

    total_time = time.time() - t1
    if total_pages > 0:
        logger.info(f"[stream] Analyze time: {total_time:.2f}s, pages: {total_pages}, throughput: {total_pages/total_time:.1f} pages/s")
    logger.info(f"[stream] Done {len(ordered_results)} PDFs in {time.time() - t0:.2f}s")


def process_one_pdf_file(pdf_path, save_dir=None, lang="ch"):
    if not save_dir:
        save_dir = "/paratera_ningxia/user/zhangxueren/libgen_pdf_res_v2"
    pdf_file_name = os.path.basename(pdf_path)
    target_file = f"{save_dir}/{pdf_file_name}.json"
    if not os.path.exists(f"{save_dir}/"):
        os.system(f"mkdir {save_dir}/")
    #
    # if os.path.exists(target_file):
    #     print(f"the pdf result exist...[{target_file}]")
    #     return

    infer_result = infer_one_pdf(pdf_path, lang=lang)
    if infer_result is None:
        logger.warning(f"Skipping {pdf_file_name} due to processing error")
        return
        
    infer_result['pdf_path'] = pdf_path
    res_json_str = json.dumps(infer_result, ensure_ascii=False)

    with open(target_file, "w") as fo:
        fo.write(res_json_str)


def get_all_access_pdf_paths():
    print(glob.glob("/*"))
    pdf_files = glob.glob(f"/paratera_ningxia/datasets/mb-serverless-spark/zhangxueren/libgen_en_pdf/*")
    logger.info(f"{len(pdf_files)} {pdf_files[:10]}")
    print(f"{len(pdf_files)} {pdf_files[:10]}")
    book_names = set()
    with open("/paratera_ningxia/user/zhangxueren/libgen_pdf_hfw_to_process.list") as fi:
        for x in eval(fi.read()):
            book_name = os.path.basename(x)
            book_names.add(book_name)
    pdf_to_process_list = []
    for x in pdf_files:
        if os.path.basename(x) in book_names:
            pdf_to_process_list.append(x)
    random.shuffle(pdf_to_process_list)
    return pdf_to_process_list


def run_test_task():
    t0 = time.time()
    # pdf_files = glob.glob(f"/cache/lizhen/repos/MinerU/input/sample_pdf_300/*pdf")  # 使用新的testinput文件夹
    # pdf_files = sorted(pdf_files)[:100]
    pdf_files = glob.glob(f"/cache/lizhen/repos/MinerU/demo/test_pdfs/*pdf")
    # 流水线模式建议处理多份PDF
    pipeline_mode = os.environ.get("MINERU_PIPELINE_MODE", "stream").lower()
    if pipeline_mode == "serial":
        pdf_files = pdf_files[:1]
    else:
        pdf_files = pdf_files[: min(8, len(pdf_files))]
    save_dir = "/cache/lizhen/repos/mineru/MinerU/output/best_practice"
    # pdf_files = glob.glob(f"/user/zhangxueren/sample_pdf_300/*pdf")
    # save_dir = "/user/zhangxueren/sample_pdf_res"
    
    logger.info(f"Found {len(pdf_files)} documents to process")
    logger.info(f"Input directory: /cache/lizhen/repos/MinerU/demo/test_pdfs")
    logger.info(f"Output directory: {save_dir}")
    
    t1 = time.time()
    logger.info(f"Environment setup time: {t1 - t0:.2f}s")
    
    processed_count = 0
    
    if pipeline_mode == "serial":
        for idx, file_path in enumerate(pdf_files):
            if not os.path.exists(file_path):
                continue
            try:
                logger.info(f"[serial] Processing file {idx + 1}/{len(pdf_files)}: {os.path.basename(file_path)}")
                process_one_pdf_file(file_path, save_dir)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                traceback.print_stack()
                logger.error('=====' * 10)
    else:
        logger.info(f"[stream] Processing {len(pdf_files)} PDFs with streaming pipeline ...")
        infer_many_pdfs_stream(pdf_files, save_dir)
        processed_count = len(pdf_files)
    
    total_time = time.time() - t0
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Successfully processed: {processed_count}/{len(pdf_files)} files")
    logger.info(f"Average time per document: {total_time/len(pdf_files):.2f}s" if pdf_files else "No documents processed")


def main():
    import glob
    # pdf_files = glob.glob("/home/admin/zhangxueren/sample_pdf_300/*pdf")
    # random.shuffle(pdf_files)
    pdf_files = get_all_access_pdf_paths()
    print(f"pdf_files cnt:{len(pdf_files)}")
    for idx, file_path in enumerate(pdf_files):
        if not os.path.exists(file_path):
            continue

        try:
            process_one_pdf_file(file_path)
        except Exception as e:
            print(e)
            traceback.print_stack()
            print('=====' * 10)


if __name__ == "__main__":
    t0 = time.time()

    
    logger.info("Starting baseline OCR processing...")
    logger.info(f"MINERU_MODEL_SOURCE: {os.environ['MINERU_MODEL_SOURCE']}")
    logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}")
    logger.info(f"MINERU_PIPELINE_MODE: {os.environ.get('MINERU_PIPELINE_MODE','stream')}")
    
    # MineruPipelineModel(device="cuda")
    # main()
    # split_all_books()
    # copy_left_books()
    run_test_task()
    
    total_time = time.time() - t0
    logger.info(f'Total time: {total_time:.2f} seconds')
    
    # 添加运行结束标识
    logger.info("=" * 80)
    logger.info(f"BASELINE TEST RUN COMPLETED - Run ID: {run_id}")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info("=" * 80)
