# python react_code_log_dec7.py \
# --model_name claude-opus-4-5-20251101-thinking \
# --gpu_id 6



# python react_code_log_dec7.py \
# --model_name cds/Claude-4.6-opus \
# --gpu_id 6


for i in $(seq 1 50)
do
    echo "Running iteration $i..."
    python react_code_log_gen_docs.py \
    --model_name gemini-3-pro-preview \
    --gpu_id 0 \
    --config_path config/config_cross_displine_none.yaml \
    --sandbox_root "/data/yjh/react_corssdispline/none"
done