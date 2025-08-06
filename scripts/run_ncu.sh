OUTPUT_DIR="ncu_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

/usr/local/cuda-12.6/bin/ncu --set full \
    --nvtx \
    --target-processes all \
    --replay-mode kernel \
    -f -o $OUTPUT_DIR/mineru_${TIME_STAMP} \
    $PYTHON examples/ocr_pdf_with_mineru.py


# 方案1: 捕获所有内核，不使用过滤器
# echo "方案1: 捕获所有内核"
# /usr/local/cuda/bin/ncu --set full \
#   --nvtx \
#   --target-processes all \
#   --replay-mode kernel \
#   -f -o "$OUTPUT_DIR/cpmcu_kvcache_all" \
#   $PYTHON tests_nsa/test_performance/benchmark_kvcache.py

# echo "NCU profiling completed. Results saved to $OUTPUT_DIR/"
# echo "Use 'ncu-ui $OUTPUT_DIR/cpmcu_kvcache_all.ncu-rep' to view results"

# 备选方案（注释掉，需要时可启用）

# 方案2: 只在NVTX范围内捕获（如果方案1太多内核）
# echo "方案2: 使用NVTX过滤"
# /usr/local/cuda/bin/ncu --set full \
#   --nvtx \
#   --nvtx-include "flash_kvcache/" \
#   --target-processes all \
#   --replay-mode kernel \
#   -f -o "$OUTPUT_DIR/cpmcu_kvcache_nvtx" \
#   $PYTHON tests_nsa/test_performance/benchmark_kvcache.py

# 方案3: 按内核名称模糊匹配
# echo "方案3: 按内核名称匹配"
# /usr/local/cuda/bin/ncu --set full \
#   --target-processes all \
#   --replay-mode kernel \
#   --kernel-name-base "flash" \
#   -f -o "$OUTPUT_DIR/cpmcu_kvcache_flash" \
#   $PYTHON tests_nsa/test_performance/benchmark_kvcache.py

# 方案4: 轻量级分析（更快）
# echo "方案4: 轻量级分析"
# /usr/local/cuda/bin/ncu --set basic \
#   --target-processes all \
#   --kernel-name-base "fmha" \
#   -f -o "$OUTPUT_DIR/cpmcu_kvcache_basic" \
#   $PYTHON tests_nsa/test_performance/benchmark_kvcache.py
