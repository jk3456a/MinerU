import argparse
import json
import os
from typing import List

from mineru.cli.common import _process_output
from mineru.utils.enum_class import MakeMode
from mineru.data.data_reader_writer import FileBasedDataWriter


def generate_layout_pdf_from_json(input_json_path: str) -> str:
    """
    读取 run_test_task 输出的 JSON，在同一目录生成带 layout 标注的 PDF。

    要求 JSON 至少包含以下字段：
      - middle_json: 中间结构，需包含 pdf_info
      - pdf_path: 原始 PDF 文件路径（用于读取 pdf_bytes）

    返回：生成的 layout PDF 路径
    """
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        infer_result = json.load(f)

    if not isinstance(infer_result, dict):
        raise ValueError("Input JSON format invalid: expected a dict at top-level")

    middle_json = infer_result.get("middle_json")
    pdf_path = infer_result.get("pdf_path")
    model_json = infer_result.get("model_json")  # 传入以对齐 CLI，但此处不用于输出

    if middle_json is None:
        raise ValueError("JSON missing required key: 'middle_json'")
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError("JSON missing valid 'pdf_path' to read original PDF bytes")

    # 文件名：沿用 markdown 脚本逻辑，若有 pdf_path 则使用其 basename（可能包含扩展名）
    pdf_file_name = os.path.basename(pdf_path)

    # 输出目录：与 JSON 同目录
    local_md_dir = os.path.dirname(os.path.abspath(input_json_path))
    # 图片目录：接口需要，但本脚本不生成图片
    local_image_dir = os.path.join(local_md_dir, "images")

    # 读取原始 PDF 字节
    with open(pdf_path, "rb") as pf:
        pdf_bytes = pf.read()

    md_writer = FileBasedDataWriter(local_md_dir)

    pdf_info = middle_json.get("pdf_info")
    if pdf_info is None:
        raise ValueError("middle_json missing 'pdf_info'")

    # 仅生成 layout bbox 叠加 PDF
    _process_output(
        pdf_info=pdf_info,
        pdf_bytes=pdf_bytes,
        pdf_file_name=pdf_file_name,
        local_md_dir=local_md_dir,
        local_image_dir=local_image_dir,
        md_writer=md_writer,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=False,
        f_dump_orig_pdf=False,
        f_dump_md=False,
        f_dump_content_list=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_make_md_mode=MakeMode.MM_MD,
        middle_json=middle_json,
        model_output=model_json,
        is_pipeline=True,
    )

    return os.path.join(local_md_dir, f"{pdf_file_name}_layout.pdf")


def find_json_files_in_dir(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".json"):
            files.append(os.path.join(dir_path, name))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Convert MinerU JSON to layout-annotated PDF in-place (same directory)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", dest="json_file", help="单个 JSON 文件路径")
    group.add_argument("--path", dest="json_dir", help="包含多个 JSON 的目录路径")
    args = parser.parse_args()

    if args.json_file:
        out_path = generate_layout_pdf_from_json(args.json_file)
        print(f"Layout PDF 输出: {out_path}")
        return

    # 目录模式：批量处理
    json_files = find_json_files_in_dir(args.json_dir)
    if not json_files:
        print(f"目录下未找到 JSON 文件: {args.json_dir}")
        return

    for jf in json_files:
        try:
            out_path = generate_layout_pdf_from_json(jf)
            print(f"OK: {out_path}")
        except Exception as e:
            print(f"FAIL: {jf} -> {e}")


if __name__ == "__main__":
    main()


