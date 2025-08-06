OUTPUT_DIR="nsys_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MINERU_NVTX_ENABLE=true

# /usr/local/cuda-12.6/bin/nsys profile -w true \
#     -t cuda,nvtx,osrt,cudnn,cublas \
#     -s none \
#     -f true \
#     -o $OUTPUT_DIR/mineru_${TIME_STAMP} \
#     -x true \
#     $PYTHON examples/ocr_pdf_with_mineru.py

/usr/local/cuda-12.6/bin/nsys profile -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s cpu \
    -o $OUTPUT_DIR/mineru_${TIME_STAMP} \
    $PYTHON examples/ocr_pdf_with_mineru.py --use_async_transfer True

# py-spy record -o $OUTPUT_DIR/python_profile_${TIME_STAMP}.svg \
#     --duration 180 \
#     --rate 50 \
#     --subprocesses \
#     -- $PYTHON examples/ocr_pdf_with_mineru.py

# wait
# echo "Analysis complete!"