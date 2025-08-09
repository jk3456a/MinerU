OUTPUT_DIR="nsys_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

export CUDA_VISIBLE_DEVICES=2

# /usr/local/cuda-12.6/bin/nsys profile -w true \
#     -t cuda,nvtx,osrt,cudnn,cublas \
#     -s none \
#     -f true \
#     -o $OUTPUT_DIR/mineru_${TIME_STAMP} \
#     -x true \
#     $PYTHON examples/ocr_pdf_with_mineru.py

export MINERU_SKIP_TMP_IMAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MINERU_OCR_DET_MERGE_BUCKETS=0
export MINERU_MIN_BATCH_INFERENCE_SIZE=768
export MINERU_VIRTUAL_VRAM_SIZE=24
# export MINERU_OCR_DET_ONE_D2H=1
export MINERU_OCR_REC_ONE_D2H=1
export MINERU_MODEL_SOURCE=modelscope

# $PYTHON  demo/demo_stream.py
$PYTHON examples/ocr_pdf_with_mineru_baseline.py

# py-spy record -o $OUTPUT_DIR/python_profile_${TIME_STAMP}.svg \
#     --duration 180 \
#     --rate 50 \
#     --subprocesses \
#     -- $PYTHON examples/ocr_pdf_with_mineru.py

# wait
# echo "Analysis complete!"