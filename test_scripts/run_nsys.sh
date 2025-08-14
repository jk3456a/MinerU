OUTPUT_DIR="nsys_results"
mkdir -p $OUTPUT_DIR
PYTHON="/cache/lizhen/miniconda3/envs/mineru/bin/python"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

# $PYTHON  demo/demo_stream.py
/usr/local/cuda-12.6/bin/nsys profile -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s cpu \
    -o $OUTPUT_DIR/best_prectice_${TIME_STAMP} \
    $PYTHON examples/best_practice.py
