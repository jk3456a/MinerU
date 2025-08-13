import copy
import glob
import os
import random
import traceback

import json

import time
import uuid

from mineru.backend.pipeline.model_init import MineruPipelineModel
from mineru.data.data_reader_writer import FileBasedDataWriter

from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2

# 给baseline用的脚本文件，用于同步观测
# 添加日志配置
from loguru import logger

# 配置loguru日志格式
logger.remove()  # 移除默认的处理器

# 创建logs目录
os.makedirs("logs", exist_ok=True)

# 生成运行标识符
run_id = str(uuid.uuid4())[:8]  # 取前8位作为运行ID

logger.add(
    "logs/baseline_30_D2H.log",  # 日志文件
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
logger.info(f"BASELINE TEST RUN STARTED - Run ID: {run_id}")
logger.info("=" * 80)


def infer_one_pdf(pdf_file_path, lang="ch"):
    t0 = time.time()
    with open(pdf_file_path, 'rb') as fi:
        pdf_bytes = fi.read()
    t1 = time.time()
    
    try:
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
        t2 = time.time()
        logger.info(f"Read PDF file spend: {t1 - t0:.2f}s, convert PDF spend: {t2 - t1:.2f}s")
    except Exception as e:
        if "password" in str(e).lower() or "Incorrect password" in str(e):
            logger.warning(f"Skipping password-protected PDF: {os.path.basename(pdf_file_path)} - {e}")
            return None
        else:
            logger.error(f"Error processing PDF {os.path.basename(pdf_file_path)}: {e}")
            return None
    
    formula_enable = True
    table_enable = False
    pdf_name = os.path.basename(pdf_file_path)
    
    logger.info(f"Processing PDF: {pdf_name}")
    logger.info(f"Formula enable: {formula_enable}, Table enable: {table_enable}")
    
    t2_1 = time.time()
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
    logger.info(f"Total pages: {total_pages}")
    logger.info(f"Pipeline doc analyze spend: {t3 - t2_1:.2f}s")
    logger.info(f"Average time per page: {(t3 - t2_1)/total_pages:.3f}s" if total_pages > 0 else "No pages")
    logger.info(f"Processing speed: {total_pages/(t3 - t2_1):.1f} pages/s" if total_pages > 0 else "No pages")
    
    model_list = infer_results[0]
    images_list = all_image_lists[0]
    pdf_doc = all_pdf_docs[0]
    _ocr_enable = True

    model_json = copy.deepcopy(model_list)

    local_image_dir = f"/cache/lizhen/repos/mineru/MinerU/output/ocr_pdf_with_mineru_baseline/{pdf_name}"
    if not os.path.exists(local_image_dir):
        os.system(f"mkdir -p {local_image_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)

    t3_1 = time.time()
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


def process_one_pdf_file(pdf_path, save_dir=None, lang="ch"):
    if not save_dir:
        save_dir = "/paratera_ningxia/user/zhangxueren/libgen_pdf_res_v2"
    pdf_file_name = os.path.basename(pdf_path)
    target_file = f"{save_dir}/{pdf_file_name}.json"
    if not os.path.exists(f"{save_dir}/"):
        os.system(f"mkdir {save_dir}/")
    #
    if os.path.exists(target_file):
        print(f"the pdf result exist...[{target_file}]")
        return

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
    pdf_files = glob.glob(f"/cache/lizhen/repos/MinerU/input/sample_pdf_300/*pdf")  # 使用新的testinput文件夹
    #pdf按照名称排序前100篇
    pdf_files = sorted(pdf_files)[:100]
    save_dir = "/cache/lizhen/repos/mineru/MinerU/output/best_practice"
    # pdf_files = glob.glob(f"/user/zhangxueren/sample_pdf_300/*pdf")
    # save_dir = "/user/zhangxueren/sample_pdf_res"
    
    logger.info(f"Found {len(pdf_files)} documents to process")
    logger.info(f"Input directory: /cache/lizhen/repos/mineru/MinerU/demo/test_pdfs")
    logger.info(f"Output directory: {save_dir}")
    
    t1 = time.time()
    logger.info(f"Environment setup time: {t1 - t0:.2f}s")
    
    processed_count = 0
    
    for idx, file_path in enumerate(pdf_files):
        if not os.path.exists(file_path):
            continue

        try:
            logger.info(f"Processing file {idx + 1}/{len(pdf_files)}: {os.path.basename(file_path)}")
            process_one_pdf_file(file_path, save_dir)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            traceback.print_stack()
            logger.error('=====' * 10)
    
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
    # os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
    # os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "24"
    # os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = "512"
    
    logger.info("Starting baseline OCR processing...")
    logger.info(f"MINERU_MODEL_SOURCE: {os.environ['MINERU_MODEL_SOURCE']}")
    logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}")
    
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
