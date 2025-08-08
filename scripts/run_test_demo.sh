OUTPUT_DIR="nsys_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

export CUDA_VISIBLE_DEVICES=3

# /usr/local/cuda-12.6/bin/nsys profile -w true \
#     -t cuda,nvtx,osrt,cudnn,cublas \
#     -s none \
#     -f true \
#     -o $OUTPUT_DIR/mineru_${TIME_STAMP} \
#     -x true \
#     $PYTHON examples/ocr_pdf_with_mineru.py

# /usr/local/cuda-12.6/bin/nsys profile -w true \
#     -t cuda,nvtx,osrt,cudnn,cublas \
#     -s cpu \
#     -o $OUTPUT_DIR/mineru_${TIME_STAMP} \
#     $PYTHON examples/ocr_pdf_with_mineru.py

# /usr/local/cuda-12.6/bin/nsys profile -w true \
#     -t cuda,nvtx,osrt,cudnn,cublas \
#     -s cpu \
#     -o $OUTPUT_DIR/mineru_demo_${TIME_STAMP} \

$PYTHON demo/demo_test.py 

