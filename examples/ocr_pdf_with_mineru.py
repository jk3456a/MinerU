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

# ==================== 配置参数 ====================
# OCR相关配置
DEFAULT_LANG = "ch"  # 默认语言
FORMULA_ENABLE = True  # 是否启用公式识别
TABLE_ENABLE = False  # 是否启用表格识别
PARSE_METHOD = "ocr"  # 解析方法
OCR_ENABLE = True  # 是否启用OCR

os.environ['MINERU_DONOT_CLEAN_MEM'] = 'true'
os.environ['MINERU_ASYNC_DEBUG'] = 'false'
# os.environ['MINERU_ASYNC_SYNC_INTERVAL'] = '4'

PROJECT_DIR = "/cache/lizhen/repos/MinerU"

# 路径配置
DEFAULT_SAVE_DIR = PROJECT_DIR + "/output"  # 默认保存目录
LIBGEN_PDF_DIR = PROJECT_DIR + "/input/sample_pdf_300"  # PDF源文件目录
PDF_LIST_FILE = PROJECT_DIR + "/input/pdf_list.txt"  # 待处理PDF列表文件
LOCAL_IMAGE_TMP_DIR = PROJECT_DIR + "/tmp/images"  # 临时图片目录

# 测试任务路径配置
TEST_PDF_DIR = "/cache/lizhen/repos/MinerU/input/sample_pdf_300/*pdf"  # 测试PDF目录
TEST_SAVE_DIR = "/cache/lizhen/repos/MinerU/output_test"  # 测试结果保存目录

# 环境变量配置
MODEL_SOURCE = "modelscope"  # 模型源
VIRTUAL_VRAM_SIZE = "24"  # 虚拟显存大小(GB)
# MIN_BATCH_INFERENCE_SIZE = "512"  # 最小批处理推理大小（可选）

# 设备配置
DEVICE = "cuda"  # 运行设备

# 测试模式配置
TEST_MODE = True  # 是否开启测试模式（只测试1个PDF）
TEST_SINGLE_PDF = True  # 单PDF测试模式
TEST_NUM = 1


def infer_one_pdf(pdf_file_path, lang=DEFAULT_LANG):
    t0 = time.time()
    with open(pdf_file_path, 'rb') as fi:
        pdf_bytes = fi.read()
    t1 = time.time()
    new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
    t2 = time.time()
    print(f"read pdf file spend :{t1 - t0}, convert pdf spend :{t2 - t1}")
    formula_enable = FORMULA_ENABLE
    table_enable = TABLE_ENABLE
    pdf_name = os.path.basename(pdf_file_path)
    # 使用NVTX宏工具进行性能分析
    from mineru.utils.nvtx_utils import nvtx_range
    
    with nvtx_range(f"pipeline_doc_analyze: {pdf_name}"):
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                [new_pdf_bytes],
                [lang],
                parse_method=PARSE_METHOD,
                formula_enable=formula_enable, table_enable=table_enable
            )
        )
    t3 = time.time()
    print(f"pipeline_doc_analyze spend:{t3 - t2}")
    model_list = infer_results[0]
    images_list = all_image_lists[0]
    pdf_doc = all_pdf_docs[0]
    _ocr_enable = OCR_ENABLE

    model_json = copy.deepcopy(model_list)

    local_image_dir = f"{LOCAL_IMAGE_TMP_DIR}/{pdf_name}"
    if not os.path.exists(local_image_dir):
        os.system(f"mkdir -p {local_image_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)

    middle_json = pipeline_result_to_middle_json(
        model_list, images_list, pdf_doc, image_writer,
        lang, _ocr_enable, formula_enable
    )
    t4 = time.time()
    print(f"pipeline_result_to_middle_json spend:{t4 - t3}")
    print(f"sum time spend:{t4 - t0} doc_ana spend:{t3 - t2} percent:{(t3 - t2) / (t4 - t0) * 100}")
    ocr_result = {
        "middle_json": middle_json,
        "model_json": model_json
    }

    return ocr_result


def process_one_pdf_file(pdf_path, save_dir=None, lang=DEFAULT_LANG):
    if not save_dir:
        save_dir = DEFAULT_SAVE_DIR
    pdf_file_name = os.path.basename(pdf_path)
    # 修正：保存为JSON文件，避免与原PDF文件名混淆
    target_file = f"{save_dir}/{os.path.splitext(pdf_file_name)[0]}.json"
    if not os.path.exists(f"{save_dir}/"):
        os.system(f"mkdir {save_dir}/")
    # 为了测试，暂时不管有没有都执行
    # if os.path.exists(target_file):
    #     print(f"the pdf result exist...[{target_file}]")
    #     return

    infer_result = infer_one_pdf(pdf_path, lang=lang)
    infer_result['pdf_path'] = pdf_path
    res_json_str = json.dumps(infer_result, ensure_ascii=False, indent=2)

    with open(target_file, "w", encoding='utf-8') as fo:
        fo.write(res_json_str)
    
    print(f"✅ Pipeline处理完成，结果保存到: {target_file}")


def get_all_access_pdf_paths():
    print(glob.glob("/*"))
    pdf_files = glob.glob(LIBGEN_PDF_DIR + "/*pdf")  # 修正路径
    print(f"{len(pdf_files)} {pdf_files[:10]}")
    book_names = set()
    with open(PDF_LIST_FILE) as fi:
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
    pdf_files = glob.glob(TEST_PDF_DIR)
    save_dir = TEST_SAVE_DIR
    
    print("=" * 80)
    print("MinerU Pipeline后端性能测试")
    print("=" * 80)
    
    if TEST_MODE and TEST_SINGLE_PDF:
        # 测试模式：只测试第一个PDF
        if pdf_files:
            pdf_files = pdf_files[:TEST_NUM]
            print(f"🔬 测试模式：只测试{TEST_NUM}个PDF文件")
        else:
            print("❌ 未找到PDF文件")
            return
    
    print(f"待测试PDF文件数量: {len(pdf_files)}")
    if pdf_files:
        test_file = pdf_files[0]
        file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
        print(f"测试文件: {os.path.basename(test_file)} ({file_size:.2f} MB)")
    
    success_count = 0
    failed_count = 0
    total_start_time = time.time()
    
    for idx, file_path in enumerate(pdf_files):
        if not os.path.exists(file_path):
            continue

        print(f"\n📄 处理进度: {idx + 1}/{len(pdf_files)} - {os.path.basename(file_path)}")
        
        try:
            start_time = time.time()
            process_one_pdf_file(file_path, save_dir)
            process_time = time.time() - start_time
            print(f"✅ Pipeline处理耗时: {process_time:.2f}秒")
            success_count += 1
                
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            traceback.print_exc()
            print('=====' * 10)
            failed_count += 1

    total_time = time.time() - total_start_time
    print(f"\n" + "=" * 80)
    print(f"Pipeline后端测试完成!")
    print(f"成功: {success_count}, 失败: {failed_count}")
    print(f"总耗时: {total_time:.2f}秒")
    if success_count > 0 and 'file_size' in locals():
        avg_throughput = file_size * success_count / total_time
        print(f"平均处理速度: {avg_throughput:.2f} MB/s")
    print("=" * 80)


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
    os.environ["MINERU_MODEL_SOURCE"] = MODEL_SOURCE
    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = VIRTUAL_VRAM_SIZE
    # os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = MIN_BATCH_INFERENCE_SIZE
    start_time = time.time()
    # MineruPipelineModel(device=DEVICE)
    # main()
    # split_all_books()
    # copy_left_books()
    run_test_task()
    total_time = time.time() - start_time
    print(f'Total time: {total_time:.2f} seconds')
