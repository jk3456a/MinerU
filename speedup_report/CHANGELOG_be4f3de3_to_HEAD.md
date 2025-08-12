# 从 be4f3de3 到 HEAD 的变更摘要

## 概览
- 范围：be4f3de3（Update version.py with new version）→ HEAD
- 总计：19 文件变更，+1409 / -188

## 主要新增
- 流式管线：`mineru/backend/pipeline/pipeline_analyze_stream.py` (已废弃 持有多个pdf可能存在泄露风险，且加速效果不明显)
  - 为分析流程提供流式处理/输出能力（便于长任务可观测与增量产出）。
- 性能标注工具：`mineru/utils/nvtx_utils.py`
  - 提供 NVTX 标注，配合 `scripts/run_nsys.sh` 进行 Nsight Systems 性能分析。
- 示例与脚本：
  - `examples/best_prectice.py` 🏆关键脚本，实现主要功能，给定pdf，调用mineru的接口生成文本内容的json
  - `examples/json_to_markdown.py` 给定生成的json，调用mineru的接口生成文本内容的markdown

## 关键改动
- Pipeline（占比 21%）
  - `batch_analyze.py`、`pipeline_analyze.py`、`model_json_to_middle_json.py`：重构/增强批处理与管线调度、模型中间结果转换逻辑，提升稳定性与扩展性。
- Utils（占比 21%）
  - `pdf_image_tools.py`、`pdf_reader.py`、`cut_image.py`：图像/PDF 处理路径优化，增强鲁棒性与性能。
- OCR 推理与后处理（合计 ~21%）
  - `paddleocr2pytorch/tools/infer/predict_{det,rec}.py`、`pytorchocr/postprocess/rec_postprocess.py`、`pytorch_paddle.py`：检测/识别流程与后处理改进，阈值与 I/O 兼容性更优，仅修改了默认路径
- 模型与布局
  - `model/layout/doclayout_yolo.py`：布局检测细节优化。
  - `model/mfr/unimernet/Unimernet.py`：接口与推理稳定性改良。
- CLI
  - `mineru/cli/common.py`：命令行入口的小幅增强与参数兼容性处理。

## 对外接口与兼容性
- 新增流式分析能力可能引入新的可选参数/回调；原有批处理接口保持可用。
- OCR 推理与后处理的内部调整对默认行为更稳健，期望向后兼容；如有自定义脚本依赖内部实现细节，需回归测试。
- 新增 NVTX 不影响功能使用；仅在启用性能分析脚本时生效。

## 性能与工具
- 配套 `scripts/run_nsys.sh` 进行全链路性能剖析，结合 `mineru/utils/nvtx_utils.py` 的 NVTX 标注，可快速定位瓶颈。
- Pipeline 与图像/PDF 工具的批量路径优化，预计降低 I/O 与预处理阶段耗时。
