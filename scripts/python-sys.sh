#!/bin/bash
OUTPUT_DIR="nsys_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
FLAG="examples/ocr_pdf_with_mineru.py"


CUDA_DEVICE=0,1,2,3

echo "Starting focused analysis..."

# 1. 运行分析
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE py-spy record -o $OUTPUT_DIR/python_profile_${TIME_STAMP}.svg \
    --duration 180 \
    --rate 50 \
    -- $PYTHON $FLAG

