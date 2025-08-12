import argparse
import json
import os
from typing import Optional, List

from mineru.cli.common import _process_output
from mineru.utils.enum_class import MakeMode
from mineru.data.data_reader_writer import FileBasedDataWriter


def generate_markdown_from_json(
    input_json_path: str,
    make_md_mode: str = MakeMode.MM_MD,
) -> str:
    """
    读取 run_test_task 输出的 JSON，使用 common._process_output 在同一目录生成 Markdown。

    参数:
        input_json_path: JSON 文件路径（内容包含 middle_json, model_json, pdf_path）
        make_md_mode: Markdown 生成模式，默认 MakeMode.MM_MD

    返回:
        生成的 Markdown 文件路径（与 JSON 同目录）
    """
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        infer_result = json.load(f)

    if not isinstance(infer_result, dict):
        raise ValueError("Input JSON format invalid: expected a dict at top-level")

    middle_json = infer_result.get("middle_json")
    model_json = infer_result.get("model_json")
    pdf_path = infer_result.get("pdf_path")

    if middle_json is None or model_json is None:
        raise ValueError("JSON missing required keys: 'middle_json' and/or 'model_json'")

    if pdf_path and os.path.exists(pdf_path):
        pdf_file_name = os.path.basename(pdf_path)
    else:
        base_name = os.path.basename(input_json_path)
        pdf_file_name = base_name[:-5] if base_name.endswith(".json") else base_name

    # 输出目录：与 JSON 同目录
    local_md_dir = os.path.dirname(os.path.abspath(input_json_path))
    # 图片目录：与 Markdown 同目录下的 images 子目录（Markdown 中将引用 images/...）
    local_image_dir = os.path.join(local_md_dir, "images")

    md_writer = FileBasedDataWriter(local_md_dir)

    # 仅生成 Markdown（不生成 bbox、原始 PDF、副产物）
    pdf_info = middle_json.get("pdf_info")
    if pdf_info is None:
        raise ValueError("middle_json missing 'pdf_info'")

    _process_output(
        pdf_info=pdf_info,
        pdf_bytes=b"",  # 不需要原始 PDF 字节
        pdf_file_name=pdf_file_name,
        local_md_dir=local_md_dir,
        local_image_dir=local_image_dir,
        md_writer=md_writer,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_orig_pdf=False,
        f_dump_md=True,
        f_dump_content_list=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_make_md_mode=make_md_mode,
        middle_json=middle_json,
        model_output=model_json,
        is_pipeline=True,
    )

    return os.path.join(local_md_dir, f"{pdf_file_name}.md")


def find_json_files_in_dir(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".json"):
            files.append(os.path.join(dir_path, name))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Convert MinerU JSON to Markdown in-place (same directory)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", dest="json_file", help="单个 JSON 文件路径")
    group.add_argument("--path", dest="json_dir", help="包含多个 JSON 的目录路径")
    parser.add_argument(
        "--mode",
        dest="mode",
        default=MakeMode.MM_MD,
        choices=[MakeMode.MM_MD, MakeMode.NLP_MD, MakeMode.CONTENT_LIST],
        help="Markdown 生成模式",
    )

    args = parser.parse_args()

    if args.json_file:
        md_path = generate_markdown_from_json(
            input_json_path=args.json_file,
            make_md_mode=args.mode,
        )
        print(f"Markdown 输出: {md_path}")
        return

    # 目录模式：批量处理
    json_files = find_json_files_in_dir(args.json_dir)
    if not json_files:
        print(f"目录下未找到 JSON 文件: {args.json_dir}")
        return

    for jf in json_files:
        try:
            md_path = generate_markdown_from_json(jf, make_md_mode=args.mode)
            print(f"OK: {md_path}")
        except Exception as e:
            print(f"FAIL: {jf} -> {e}")


if __name__ == "__main__":
    main()
