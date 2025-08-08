# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
import time
from pathlib import Path

from loguru import logger

# 配置loguru日志格式
logger.remove()  # 移除默认的处理器

# 创建logs目录
os.makedirs("logs", exist_ok=True)

# 生成运行标识符
import uuid
run_id = str(uuid.uuid4())[:8]  # 取前8位作为运行ID

logger.add(
    "logs/demo_stream_5.log",  # 日志文件
    format="\n{time:YYYY-MM-DD HH:mm:ss} | {level} | [{run_id}] {message}",
    level="INFO",
    rotation="10 MB",  # 日志文件大小超过10MB时轮转
    retention="7 days",  # 保留7天的日志
    filter=lambda record: record.update(run_id=run_id) or True
)
logger.add(
    lambda msg: print(msg, end=""),  # 同时输出到控制台
    format="{time:HH:mm:ss} | {level} | [{run_id}] {message}",
    level="INFO",
    filter=lambda record: record.update(run_id=run_id) or True
)

# 添加运行开始标识
logger.info("=" * 80)
logger.info(f"DEMO TEST RUN STARTED - Run ID: {run_id}")
logger.info("=" * 80)

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze_stream import doc_analyze_stream
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.pdf_image_tools import load_images_from_pdf
import pypdfium2 as pdfium


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="ocr",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=False,  # Enable table parsing
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        t0 = time.time()
        # 1) 预处理与过滤（保持与原逻辑一致）
        valid_pdf_bytes_list: list[bytes] = []
        valid_pdf_file_names: list[str] = []
        valid_p_lang_list: list[str] = []
        skipped_count = 0

        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            try:
                new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
                valid_pdf_bytes_list.append(new_pdf_bytes)
                valid_pdf_file_names.append(pdf_file_names[idx])
                valid_p_lang_list.append(p_lang_list[idx])
            except Exception as e:
                if "password" in str(e).lower() or "incorrect password" in str(e).lower():
                    logger.warning(f"Skipping password-protected PDF: {pdf_file_names[idx]} - {e}")
                    skipped_count += 1
                else:
                    logger.error(f"Error processing PDF {pdf_file_names[idx]}: {e}")
                    skipped_count += 1

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} PDF files due to errors (including password protection)")

        if not valid_pdf_bytes_list:
            logger.error("No valid PDF files to process after filtering")
            return

        t1 = time.time()
        logger.info(f"Convert PDF spend: {t1 - t0:.2f}s (processed {len(valid_pdf_bytes_list)}/{len(pdf_bytes_list)} files)")

        # 2) 预先获取每个PDF的页数，用于确定何时完成一个PDF的聚合
        page_count_map: dict[int, int] = {}
        for pdf_idx, pdf_bytes in enumerate(valid_pdf_bytes_list):
            try:
                doc = pdfium.PdfDocument(pdf_bytes)
                page_count_map[pdf_idx] = len(doc)
                doc.close()
            except Exception as e:
                logger.warning(f"Failed to open PDF for page count, will skip index {pdf_idx}: {e}")
                page_count_map[pdf_idx] = 0

        # 3) 启动流式分析并聚合结果到每个PDF
        pdf_model_pages: dict[int, dict[int, dict]] = {}
        page_received_count: dict[int, int] = {}
        total_pages_processed = 0

        # 约束：当前parse_method默认'ocr'，因此OCR启用
        _parse_method = parse_method

        t2 = time.time()
        for res in doc_analyze_stream(
            valid_pdf_bytes_list,
            valid_p_lang_list,
            parse_method=_parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
        ):
            pdf_idx = res["pdf_idx"]
            page_idx = res["page_idx"]
            layout_dets = res["layout_dets"]
            page_info = res["page_info"]

            if pdf_idx not in pdf_model_pages:
                pdf_model_pages[pdf_idx] = {}

            pdf_model_pages[pdf_idx][page_idx] = {
                "layout_dets": layout_dets,
                "page_info": page_info,
            }

            page_received_count[pdf_idx] = page_received_count.get(pdf_idx, 0) + 1
            total_pages_processed += 1

            # 在流式推理期间先不落盘，避免与PDF渲染并发导致底层库线程不安全
            # 推理完成后统一聚合并落盘

        t3 = time.time()
        logger.info(f"Pipeline doc analyze (stream) spend: {t3 - t2:.2f}s")
        if total_pages_processed > 0 and (t3 - t2) > 0:
            logger.info(
                f"Average time per page: {(t3 - t2)/total_pages_processed:.3f}s, "
                f"Processing speed: {total_pages_processed/(t3 - t2):.1f} pages/s"
            )
        logger.info(f"Total PDFs processed: {len(valid_pdf_file_names)}")

        # 4) 推理结束后，逐PDF聚合与落盘（避免流式期间PDFium并发）
        t4 = time.time()
        for pdf_idx in sorted(pdf_model_pages.keys()):
            page_cnt = page_count_map.get(pdf_idx, 0)
            model_list = [pdf_model_pages[pdf_idx][i] for i in range(page_cnt)] if page_cnt > 0 else []
            pdf_file_name = valid_pdf_file_names[pdf_idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, _parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list, pdf_doc = load_images_from_pdf(valid_pdf_bytes_list[pdf_idx], dpi=200)
            _lang = valid_p_lang_list[pdf_idx]
            _ocr_enable = True if _parse_method == "ocr" else False

            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable
            )

            ocr_result = {
                "middle_json": middle_json,
                "model_json": copy.deepcopy(model_list),
            }

            target_file = f"{local_md_dir}/{pdf_file_name}.json"
            res_json_str = json.dumps(ocr_result, ensure_ascii=False)
            with open(target_file, "w") as fo:
                fo.write(res_json_str)

            logger.info(
                f"Saved {pdf_file_name} ({len(model_list)} pages) to {local_md_dir}"
            )

        # 释放内存
        pdf_model_pages.clear()
        page_received_count.clear()
        logger.info(f"Pipeline result to middle json (stream aggregate) spend: {time.time() - t4:.2f}s")

def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="ocr",
        start_page_id=0,
        end_page_id=None,
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    t0 = time.time()
    # args
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    # pdf_files_dir = os.path.join(__dir__, "testinput")  # 使用新的testinput文件夹
    # output_dir = os.path.join(__dir__, "testoutput")
    pdf_files_dir = os.path.join(__dir__, "test_pdfs")
    output_dir = os.path.join(__dir__, "test_output")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    logger.info(f"Found {len(doc_path_list)} documents to process")
    logger.info(f"Input directory: {pdf_files_dir}")
    logger.info(f"Output directory: {output_dir}")

    """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "24"
    os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = "768"
    t1 = time.time()
    logger.info(f"Environment setup time: {t1 - t0:.2f}s")
    

    """Use pipeline mode if your environment does not support VLM"""
    parse_doc(doc_path_list, output_dir, backend="pipeline")

    """To enable VLM mode, change the backend to 'vlm-xxx'"""
    # parse_doc(doc_path_list, output_dir, backend="vlm-transformers")
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-engine")
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-client", server_url="http://127.0.0.1:30000")
    
    total_time = time.time() - t0
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Average time per document: {total_time/len(doc_path_list):.2f}s" if doc_path_list else "No documents processed")
    
    # 添加运行结束标识
    logger.info("=" * 80)
    logger.info(f"DEMO TEST RUN COMPLETED - Run ID: {run_id}")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info("=" * 80)