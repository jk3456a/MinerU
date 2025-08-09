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
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "24"

# 路径配置
PROJECT_DIR = "/cache/lizhen/repos/MinerU"
DEFAULT_SAVE_DIR = PROJECT_DIR + "/output"  # 默认保存目录
LIBGEN_PDF_DIR = PROJECT_DIR + "/input/sample_pdf_300"  # PDF源文件目录
PDF_LIST_FILE = PROJECT_DIR + "/input/pdf_list.txt"  # 待处理PDF列表文件
LOCAL_IMAGE_TMP_DIR = PROJECT_DIR + "/tmp/images"  # 临时图片目录

# 测试模式配置
TEST_MODE = True  # 是否开启测试模式（只测试1个PDF）
TEST_SINGLE_PDF = True  # 单PDF测试模式
TEST_NUM = 20
# ==================== 配置参数 ====================

# input: 
# 1. 待处理的pdf列表
# 2. 待处理的pdf列表的标签
# output:
# 1. 处理后的pdf列表
# 2. 处理后的pdf列表的标签
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



def get_all_access_pdf_paths():
