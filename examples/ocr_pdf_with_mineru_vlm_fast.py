import copy
import glob
import os
import random
import traceback

import json

import time
import uuid

from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.predictor import get_predictor
from mineru.data.data_reader_writer import FileBasedDataWriter

# ==================== 配置参数 ====================
# VLM相关配置
DEFAULT_LANG = "ch"  # 默认语言
FORMULA_ENABLE = True  # 是否启用公式识别
TABLE_ENABLE = True  # 是否启用表格识别

PROJECT_DIR = "/cache/lizhen/repos/MinerU"

# 路径配置
DEFAULT_SAVE_DIR = PROJECT_DIR + "/output_vlm"  # VLM结果保存目录
LIBGEN_PDF_DIR = PROJECT_DIR + "/input/sample_pdf_300"  # PDF源文件目录
PDF_LIST_FILE = PROJECT_DIR + "/input/pdf_list.txt"  # 待处理PDF列表文件
LOCAL_IMAGE_TMP_DIR = PROJECT_DIR + "/tmp/images_vlm"  # 临时图片目录

# 测试任务路径配置
TEST_PDF_DIR = "/cache/lizhen/repos/MinerU/input/sample_pdf_300/*pdf"  # 测试PDF目录
TEST_SAVE_DIR = "/cache/lizhen/repos/MinerU/output_vlm"  # 测试结果保存目录

# VLM后端配置
VLM_BACKEND = "sglang-engine"  # 可选: "sglang-engine", "sglang-client", "transformers"
# VLM_MODEL_PATH = "opendatalab/MinerU2.0-2505-0.9B"  # HuggingFace
VLM_MODEL_PATH = "OpenDataLab/MinerU2.0-2505-0.9B"  # ModelScope
SGLANG_SERVER_URL = "http://localhost:30000/v1/chat/completions"  # sglang-client时使用

# 环境变量配置
# MODEL_SOURCE = "huggingface"  # 如果要从HuggingFace下载，使用"huggingface"
MODEL_SOURCE = "modelscope"  # 如果要从ModelScope下载，使用"modelscope"
VIRTUAL_VRAM_SIZE = "24"  # 虚拟显存大小(GB)

# 设备配置
DEVICE = "cuda"  # 运行设备

# 测试模式配置
TEST_MODE = False  # 是否开启测试模式（只测试1个PDF）
TEST_SINGLE_PDF = False  # 单PDF测试模式

# VLM性能优化参数
# VLM_PERFORMANCE_CONFIG = {
#     "temperature": 0.0,
#     "top_p": 0.9,
#     "top_k": 50,
#     "max_new_tokens": 4096,
#     "repetition_penalty": 1.0,
#     "presence_penalty": 0.0,
#     # SGLang性能优化参数
#     "enable_torch_compile": True,  # 启用torch compile约15%加速
#     "tp_size": 1,  # 张量并行，多卡时可增加
#     "dp_size": 1,  # 数据并行，多卡时可增加
#     "mem_fraction_static": 0.7,  # KV缓存大小，显存不足时降低
# }

# 全局predictor缓存
_global_predictor = None


def get_vlm_predictor():
    """获取VLM预测器，使用全局缓存避免重复初始化"""
    global _global_predictor
    
    if _global_predictor is None:
        print("初始化VLM预测器...")
        start_time = time.time()
        
        if VLM_BACKEND == "sglang-engine":
            # 使用sglang-engine后端（本地推理，最快）
            _global_predictor = get_predictor(
                backend="sglang-engine",
                model_path=VLM_MODEL_PATH,
                # **VLM_PERFORMANCE_CONFIG
            )
        elif VLM_BACKEND == "sglang-client":
            # 使用sglang-client后端（连接sglang-server）
            _global_predictor = get_predictor(
                backend="sglang-client",
                server_url=SGLANG_SERVER_URL,
                # temperature=VLM_PERFORMANCE_CONFIG["temperature"],
                # top_p=VLM_PERFORMANCE_CONFIG["top_p"],
                # top_k=VLM_PERFORMANCE_CONFIG["top_k"],
                # max_new_tokens=VLM_PERFORMANCE_CONFIG["max_new_tokens"],
                # repetition_penalty=VLM_PERFORMANCE_CONFIG["repetition_penalty"],
                # presence_penalty=VLM_PERFORMANCE_CONFIG["presence_penalty"],
            )
        elif VLM_BACKEND == "transformers":
            # 使用transformers后端（兼容性最好但较慢）
            _global_predictor = get_predictor(
                backend="transformers",
                model_path=VLM_MODEL_PATH,
                # temperature=VLM_PERFORMANCE_CONFIG["temperature"],
                # top_p=VLM_PERFORMANCE_CONFIG["top_p"],
                # top_k=VLM_PERFORMANCE_CONFIG["top_k"],
                # max_new_tokens=VLM_PERFORMANCE_CONFIG["max_new_tokens"],
                # repetition_penalty=VLM_PERFORMANCE_CONFIG["repetition_penalty"],
                # presence_penalty=VLM_PERFORMANCE_CONFIG["presence_penalty"],
            )
        else:
            raise ValueError(f"不支持的VLM后端: {VLM_BACKEND}")
        
        init_time = time.time() - start_time
        print(f"VLM预测器初始化完成，耗时: {init_time:.2f}秒")
    
    return _global_predictor


def infer_one_pdf_vlm(pdf_file_path, lang=DEFAULT_LANG):
    """使用VLM后端处理单个PDF文件"""
    t0 = time.time()
    
    # 读取PDF文件
    with open(pdf_file_path, 'rb') as fi:
        pdf_bytes = fi.read()
    t1 = time.time()
    print(f"读取PDF文件耗时: {t1 - t0:.3f}秒")
    
    # 获取VLM预测器
    predictor = get_vlm_predictor()
    t2 = time.time()
    print(f"获取预测器耗时: {t2 - t1:.3f}秒")
    
    # 设置图片保存目录
    pdf_name = os.path.basename(pdf_file_path)
    local_image_dir = f"{LOCAL_IMAGE_TMP_DIR}/{pdf_name}"
    if not os.path.exists(local_image_dir):
        os.makedirs(local_image_dir, exist_ok=True)
    image_writer = FileBasedDataWriter(local_image_dir)
    
    # 使用VLM后端进行文档分析
    print(f"开始VLM分析，后端: {VLM_BACKEND}")
    middle_json, vlm_results = vlm_doc_analyze(
        pdf_bytes,
        image_writer=image_writer,
        predictor=predictor,
        backend=VLM_BACKEND,
        # 可以传递额外的VLM参数
        formula_enable=FORMULA_ENABLE,
        table_enable=TABLE_ENABLE,
    )
    t3 = time.time()
    print(f"VLM文档分析耗时: {t3 - t2:.3f}秒")
    
    # 构造结果
    vlm_result = {
        "middle_json": middle_json,
        "vlm_results": vlm_results,
        "backend": VLM_BACKEND,
        "model_path": VLM_MODEL_PATH,
        # "performance_config": VLM_PERFORMANCE_CONFIG
    }
    
    total_time = t3 - t0
    analysis_percent = (t3 - t2) / total_time * 100
    print(f"总耗时: {total_time:.3f}秒, VLM分析占比: {analysis_percent:.1f}%")
    
    return vlm_result


def process_one_pdf_file(pdf_path, save_dir=None, lang=DEFAULT_LANG):
    """处理单个PDF文件，保持原有逻辑不变"""
    if not save_dir:
        save_dir = DEFAULT_SAVE_DIR
    
    pdf_file_name = os.path.basename(pdf_path)
    target_file = f"{save_dir}/{pdf_file_name}.json"  # 添加.json后缀便于区分
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 检查结果是否已存在
    if os.path.exists(target_file):
        print(f"VLM结果已存在: [{target_file}]")
        return
    
    try:
        # 使用VLM后端处理
        print(f"开始处理PDF: {pdf_file_name}")
        infer_result = infer_one_pdf_vlm(pdf_path, lang=lang)
        infer_result['pdf_path'] = pdf_path
        infer_result['processing_time'] = time.time()
        
        # 保存结果
        res_json_str = json.dumps(infer_result, ensure_ascii=False, indent=2)
        with open(target_file, "w", encoding='utf-8') as fo:
            fo.write(res_json_str)
        
        print(f"VLM处理完成: {target_file}")
        
    except Exception as e:
        print(f"处理PDF失败: {pdf_file_name}, 错误: {str(e)}")
        traceback.print_exc()


def get_all_access_pdf_paths():
    """获取所有需要处理的PDF路径，保持原有逻辑"""
    print(glob.glob("/*"))
    pdf_files = glob.glob(LIBGEN_PDF_DIR + "/*pdf")  # 修正路径
    print(f"找到PDF文件: {len(pdf_files)}, 前10个: {pdf_files[:10]}")
    
    if not os.path.exists(PDF_LIST_FILE):
        print(f"PDF列表文件不存在: {PDF_LIST_FILE}, 返回所有PDF文件")
        random.shuffle(pdf_files)
        return pdf_files
    
    book_names = set()
    try:
        with open(PDF_LIST_FILE, encoding='utf-8') as fi:
            content = fi.read().strip()
            if content:
                for x in eval(content):
                    book_name = os.path.basename(x)
                    book_names.add(book_name)
    except Exception as e:
        print(f"读取PDF列表文件失败: {e}, 返回所有PDF文件")
        random.shuffle(pdf_files)
        return pdf_files
    
    pdf_to_process_list = []
    for x in pdf_files:
        if os.path.basename(x) in book_names:
            pdf_to_process_list.append(x)
    
    random.shuffle(pdf_to_process_list)
    print(f"根据列表筛选后的PDF文件数: {len(pdf_to_process_list)}")
    return pdf_to_process_list


def run_test_task():
    """运行测试任务，保持原有逻辑"""
    pdf_files = glob.glob(TEST_PDF_DIR)
    save_dir = TEST_SAVE_DIR
    
    print("=" * 80)
    print("MinerU VLM高速后端性能测试")
    print(f"后端: {VLM_BACKEND}")
    print("=" * 80)
    
    if TEST_MODE and TEST_SINGLE_PDF:
        # 测试模式：只测试第一个PDF
        if pdf_files:
            pdf_files = pdf_files[:1]
            print(f"🔬 测试模式：只测试1个PDF文件")
        else:
            print("❌ 未找到PDF文件")
            return
    
    print(f"待测试PDF文件数量: {len(pdf_files)}")
    if pdf_files:
        test_file = pdf_files[0]
        file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
        print(f"测试文件: {os.path.basename(test_file)} ({file_size:.2f} MB)")
    
    # 预初始化VLM预测器
    print(f"\n🚀 初始化{VLM_BACKEND}预测器...")
    init_start_time = time.time()
    try:
        get_vlm_predictor()
        init_time = time.time() - init_start_time
        print(f"✅ 预测器初始化完成，耗时: {init_time:.2f}秒")
    except Exception as e:
        print(f"❌ VLM预测器初始化失败: {e}")
        print("💡 建议: 1) 检查显存是否充足 2) 尝试使用transformers后端")
        return
    
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
            print(f"✅ VLM处理耗时: {process_time:.2f}秒")
            success_count += 1
            
            # 测试模式下显示详细性能信息
            if TEST_MODE:
                throughput = file_size / process_time if 'file_size' in locals() else 0
                print(f"📊 处理速度: {throughput:.2f} MB/s")
                
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            traceback.print_exc()
            print('=====' * 10)
            failed_count += 1
    
    total_time = time.time() - total_start_time
    print(f"\n" + "=" * 80)
    print(f"VLM后端测试完成!")
    print(f"后端: {VLM_BACKEND}")
    print(f"成功: {success_count}, 失败: {failed_count}")
    print(f"总耗时: {total_time:.2f}秒 (含初始化: {init_time:.2f}秒)")
    if success_count > 0 and 'file_size' in locals():
        avg_throughput = file_size * success_count / total_time
        print(f"平均处理速度: {avg_throughput:.2f} MB/s")
    print("=" * 80)


def main():
    """主函数，保持原有逻辑"""
    pdf_files = get_all_access_pdf_paths()
    print(f"主任务PDF文件数量: {len(pdf_files)}")
    
    # 预初始化VLM预测器
    try:
        get_vlm_predictor()
    except Exception as e:
        print(f"VLM预测器初始化失败: {e}")
        print("请确保已安装sglang: pip install mineru[all]")
        return
    
    success_count = 0
    failed_count = 0
    
    for idx, file_path in enumerate(pdf_files):
        if not os.path.exists(file_path):
            continue
        
        print(f"\n处理进度: {idx + 1}/{len(pdf_files)} - {os.path.basename(file_path)}")
        
        try:
            start_time = time.time()
            process_one_pdf_file(file_path)
            process_time = time.time() - start_time
            print(f"处理耗时: {process_time:.2f}秒")
            success_count += 1
            
        except Exception as e:
            print(f"处理失败: {e}")
            traceback.print_exc()
            print('=====' * 10)
            failed_count += 1
    
    print(f"\n处理完成！成功: {success_count}, 失败: {failed_count}")


def batch_process_pdfs(pdf_paths, batch_size=1):
    """批量处理PDF文件（VLM暂不支持真正的批处理，但可以优化预测器复用）"""
    print(f"批量处理 {len(pdf_paths)} 个PDF文件，批次大小: {batch_size}")
    
    # 预初始化VLM预测器
    get_vlm_predictor()
    
    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i + batch_size]
        print(f"\n处理批次 {i // batch_size + 1}: {len(batch)} 个文件")
        
        for pdf_path in batch:
            if os.path.exists(pdf_path):
                try:
                    process_one_pdf_file(pdf_path)
                except Exception as e:
                    print(f"批量处理失败: {pdf_path}, 错误: {e}")


if __name__ == "__main__":
    # 设置环境变量
    os.environ["MINERU_MODEL_SOURCE"] = MODEL_SOURCE  # 设置模型源
    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = VIRTUAL_VRAM_SIZE
    
    # SGLang性能优化环境变量
    if VLM_BACKEND.startswith("sglang"):
        # 启用torch编译加速
        # if VLM_PERFORMANCE_CONFIG.get("enable_torch_compile", False):
        os.environ["SGLANG_TORCH_COMPILE"] = "1"
    
    print("=" * 60)
    print("MinerU VLM高速测试脚本")
    print(f"后端: {VLM_BACKEND}")
    print(f"模型: {VLM_MODEL_PATH}")
    print(f"设备: {DEVICE}")
    print(f"虚拟显存: {VIRTUAL_VRAM_SIZE}GB")
    print("=" * 60)
    
    start_time = time.time()
    
    # 可以选择运行测试任务或主任务
    run_test_task()  # 运行测试任务
    # main()  # 运行主任务
    
    total_time = time.time() - start_time
    print(f'\n总运行时间: {total_time:.2f} 秒')
    print("VLM高速处理完成！")