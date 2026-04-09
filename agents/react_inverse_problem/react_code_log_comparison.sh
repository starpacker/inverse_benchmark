#!/bin/bash

# Configuration
GPU_ID=6
MODEL_BASE="cds/Claude-4.6-opus_old"
# MODEL_NEW="${MODEL_BASE}-with-context"
MODEL_NEW="cds/Claude-4.6-opus"

echo "=================================================="
echo "Starting Comparison Benchmark"
echo "Original Script: react_code_log_dec7.py"
echo "New Script:      react_code_log_dec7_new.py"
echo "GPU ID:          $GPU_ID"
echo "=================================================="

# # 1. Run Original Version
# echo ""
# echo "[1/2] Running Original Version..."
# echo "Model Name: $MODEL_BASE"
# python react_code_log_dec7.py \
#     --model_name "$MODEL_BASE" \
#     --gpu_id "$GPU_ID"

# if [ $? -ne 0 ]; then
#     echo "Error: Original version failed."
#     exit 1
# fi

# 2. Run New Version
echo ""
echo "[2/2] Running New Version (with System Context)..."
echo "Model Name: $MODEL_NEW"
python react_code_log_dec7_new.py \
    --model_name "$MODEL_NEW" \
    --gpu_id "$GPU_ID"

if [ $? -ne 0 ]; then
    echo "Error: New version failed."
    exit 1
fi

echo ""
echo "=================================================="
echo "Benchmark Completed Successfully."
echo "=================================================="
echo "Generating Comparison Report..."

python compare_results.py "$MODEL_BASE" "$MODEL_NEW"

echo ""
echo "Done."
