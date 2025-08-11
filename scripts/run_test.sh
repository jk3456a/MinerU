OUTPUT_DIR="nsys_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

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
