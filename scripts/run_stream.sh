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

export MINERU_STREAM_QUEUE_SIZE=8192
export MINERU_BATCH_FLUSH_MS=1200
export MINERU_MIN_BATCH_INFERENCE_SIZE=1024

$PYTHON  demo/demo_stream.py


# py-spy record -o $OUTPUT_DIR/python_profile_${TIME_STAMP}.svg \
#     --duration 180 \
#     --rate 50 \
#     --subprocesses \
#     -- $PYTHON examples/ocr_pdf_with_mineru.py

# wait
# echo "Analysis complete!"