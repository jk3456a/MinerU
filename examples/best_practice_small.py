import copy
import glob
import os
import random
import traceback
from typing import List, Optional

# 可修改
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"  # 模型来源（如 modelscope/本地），影响下载与加载
os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "24"  # 虚拟显存(GB)，用于估算批处理比例/显存策略
os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = "768"  # 最小批推理页数，增大提升吞吐但耗内存/显存

os.environ["MINERU_SKIP_TMP_IMAGES"] = "0"  # 跳过中间表格的存储以减少I/O

# 优化的建议参数
#####################################################
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # CUDA内存分配策略，降低碎片化
os.environ["MINERU_OCR_DET_MERGE_BUCKETS"] = "0"  # 文本检测是否合并buckets(0关/1开)
os.environ["MINERU_OCR_DET_BATCH_SIZE"] = "128"  # 文本检测批大小，越大吞吐越高但显存更多
os.environ["MINERU_OCR_DET_ONE_D2H"] = "0"  # 检测阶段合并一次性D2H拷贝，减少传输
os.environ["MINERU_OCR_REC_ONE_D2H"] = "1"  # 识别阶段合并一次性D2H拷贝，减少传输


os.environ["OMP_NUM_THREADS"] = "16"  # OMP线程
os.environ["MKL_NUM_THREADS"] = "16"  # MKL线程
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # OpenBLAS线程
os.environ["MINERU_TORCH_NUM_THREADS"] = "16"  # PyTorch算子线程数
os.environ["MINERU_TORCH_NUM_INTEROP_THREADS"] = "2"  # PyTorch线程间并发度
os.environ["MINERU_OPENCV_NUM_THREADS"] = "16"  # OpenCV线程数
os.environ["MINERU_MFR_DATALOADER_WORKERS"] = "8"  # 多模态/公式识别数据加载workers
os.environ["MINERU_PDF_RENDER_WORKERS"] = "16"  # PDF渲染并发workers
os.environ["MINERU_OCR_CROP_WORKERS"] = "20"  # OCR裁剪切图并发workers
#####################################################

# 调试用，目前不开启
os.environ["MINERU_NVTX_ENABLE"] = "0"  # 启用NVTX标注，配合nsys做性能分析
os.environ["MINERU_LOG_ENABLE"] = "1"  # 启用日志输出

import json

import time
import uuid

from mineru.backend.pipeline.model_init import MineruPipelineModel
from mineru.data.data_reader_writer import FileBasedDataWriter

from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
import mineru.utils.nvtx_utils as nvtxu
from mineru.utils.logger_utils import get_logger, set_run_id

# 添加日志配置（使用项目内 logger_utils 统一风格）

_LOG_ENABLE = os.environ.get("MINERU_LOG_ENABLE", "1") == "1"

run_id = str(uuid.uuid4())[:8]  # 取前8位作为运行ID
set_run_id(run_id)
logger = get_logger("best_practice_small", file_name="best_prectice_small.log")

# 添加运行开始标识
logger.info("=" * 80)
logger.info(f"BASELINE TEST RUN STARTED - Run ID: {run_id}")
logger.info("=" * 80)

def infer_n_pdf(pdf_paths, lang="ch", parallel_num=50):
    """批量推理n个PDF文件"""
    pdf_bytes_list = []
    file_names = []
    
    # 1. 读取和转换PDF文件
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as fi:
                pdf_bytes = fi.read()
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
            pdf_bytes_list.append(new_pdf_bytes)
            file_names.append(os.path.basename(pdf_path))
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            continue
    
    if not pdf_bytes_list:
        logger.warning("No valid PDF files to process")
        return []
    
    # 2. 批量推理
    formula_enable = True
    table_enable = True
    lang_list = [lang] * len(pdf_bytes_list)
    
    try:
        logger.info(f"Starting batch inference for {len(pdf_bytes_list)} PDFs")
        with nvtxu.nvtx_range(f"batch_pipeline_doc_analyze_{len(pdf_bytes_list)}_pdfs"):
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
                pipeline_doc_analyze(
                    pdf_bytes_list,
                    lang_list,
                    parse_method="ocr",
                    formula_enable=formula_enable,
                    table_enable=table_enable
                )
            )
    except Exception as e:
        logger.error(f"Batch pipeline analysis failed: {e}")
        return []
    
    # 3. 后处理 - 生成middle_json
    results = []
    for idx, (model_list, images_list, pdf_doc) in enumerate(zip(
        infer_results, all_image_lists, all_pdf_docs)):
        
        file_name = file_names[idx]
        pdf_path = pdf_paths[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        
        # 创建图像写入器
        local_image_dir = f"/cache/lizhen/repos/mineru/MinerU/output/best_practice/{file_name}"
        if not os.path.exists(local_image_dir):
            os.system(f"mkdir -p {local_image_dir}")
        image_writer = FileBasedDataWriter(local_image_dir)
        
        try:
            with nvtxu.nvtx_range(f"middle_json_{file_name}"):
                middle_json = pipeline_result_to_middle_json(
                    model_list, images_list, pdf_doc, image_writer,
                    _lang, _ocr_enable, formula_enable
                )
            
            # 计算页数
            total_pages = len(images_list) if images_list else 0
            
            result = {
                "pdf_path": pdf_path,
                "file_name": file_name,
                "middle_json": middle_json,
                "model_json": copy.deepcopy(model_list),
                "total_pages": total_pages,
                "lang": _lang
            }
            
            results.append(result)
            logger.info(f"Processed {file_name}: {total_pages} pages")
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            continue
    
    logger.info(f"Batch inference completed: {len(results)}/{len(pdf_paths)} successful")
    return results
    

def process_n_pdf_files(pdf_paths, save_dir, parallel_num=50, lang="ch"):
    """批量处理n个PDF文件并保存结果"""
    # 过滤存在的文件
    pdf_files = [x for x in pdf_paths if os.path.exists(x)]
    
    if not pdf_files:
        logger.warning("No valid PDF files found")
        return 0
    
    logger.info(f"Processing {len(pdf_files)} PDF files in batch")
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.system(f"mkdir -p {save_dir}")
    
    # 批量推理
    infer_results = infer_n_pdf(pdf_files, lang=lang, parallel_num=parallel_num)
    if not infer_results:
        logger.warning(f"No results from batch inference")
        return 0
    
    # 保存结果 - 与best_practice.py保持一致
    saved_count = 0
    for infer_result in infer_results:
        try:
            file_name = infer_result["file_name"]
            target_file = f"{save_dir}/{file_name}.json"
            
            # 只保存核心数据，与best_practice.py一致
            core_result = {
                "middle_json": infer_result["middle_json"],
                "model_json": infer_result["model_json"]
            }
            core_result['pdf_path'] = infer_result["pdf_path"]
            
            res_json_str = json.dumps(core_result, ensure_ascii=False)
            
            with open(target_file, "w") as fo:
                fo.write(res_json_str)
                
            saved_count += 1
            logger.info(f"Saved result for {file_name} ({infer_result['total_pages']} pages)")
            
        except Exception as e:
            logger.error(f"Error saving {infer_result.get('file_name', 'unknown')}: {e}")
            continue
    
    logger.info(f"Successfully processed and saved {saved_count}/{len(pdf_files)} files")
    return saved_count

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
    # pdf的平均页数，根据实际测试调整
    ave_pdf_pages = 15
    parallel_num = int(os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"]) // ave_pdf_pages - 1
    t0 = time.time()
    # pdf_files = glob.glob(f"/cache/lizhen/repos/MinerU/input/sample_pdf_300/*pdf")  # 使用新的testinput文件夹
    # pdf_files = sorted(pdf_files)[:100]
    pdf_files = glob.glob(f"/cache/lizhen/repos/mineru/MinerU/testdata/qikan_pdf_sample/*pdf")
    #测试少量文件
    pdf_files = pdf_files[:100] 
    save_dir = "/cache/lizhen/repos/mineru/MinerU/output/best_practice"
    # pdf_files = glob.glob(f"/user/zhangxueren/sample_pdf_300/*pdf")
    # save_dir = "/user/zhangxueren/sample_pdf_res"
    
    logger.info(f"Found {len(pdf_files)} documents to process")
    logger.info(f"Input directory: /cache/lizhen/repos/mineru/MinerU/testdata/qikan_pdf_sample")
    logger.info(f"Output directory: {save_dir}")
    
    if not pdf_files:
        logger.warning("No PDF files found to process")
        return
    
    t1 = time.time()
    logger.info(f"Environment setup time: {t1 - t0:.2f}s")
    
    processed_count = 0
    
    for i in range(0, len(pdf_files), parallel_num):
        batch_files = pdf_files[i:i+parallel_num]
        batch_idx = i // parallel_num + 1
        logger.info(f"Processing batch {batch_idx}: {len(batch_files)} files")
        
        try:
            batch_processed = process_n_pdf_files(batch_files, save_dir, parallel_num=parallel_num)
            processed_count += batch_processed
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            traceback.print_exc()
    
    total_time = time.time() - t0
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Successfully processed: {processed_count}/{len(pdf_files)} files")
    logger.info(f"Average time per document: {total_time/len(pdf_files):.2f}s" if pdf_files else "No documents processed")

if __name__ == "__main__":
    t0 = time.time()

    
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
