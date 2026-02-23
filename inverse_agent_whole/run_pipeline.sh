export CUDA_VISIBLE_DEVICES=7

# 设置输出目录
OUTPUT_DIR="results_pipeline_gemini"

# 1. Spectral CT
# python pipeline.py \
#   --task_name "spectral_ct_examples-master" \
#   --gt_code "/data/yjh/spectral_ct_examples-master_sandbox/spectral_ct_code.py" \
#   --working_folder "/data/yjh/pipeline_work" \
#   --output_dir "$OUTPUT_DIR" \
#   --model "gemini-3-pro-preview"

# # 2. STORM Analysis
# python pipeline.py \
#   --task_name "storm_analysis-master" \
#   --gt_code "/data/yjh/storm-analysis-master_sandbox/storm_code.py" \
#   --working_folder "/data/yjh/pipeline_work" \
#   --output_dir "$OUTPUT_DIR" \
#   --model "gemini-3-pro-preview"

# # 3. BPM (Beam Propagation Method)
# python pipeline.py \
#   --task_name "bpm" \
#   --gt_code "/home/yjh/bpm_sandbox/bpm_code.py" \
#   --working_folder "/data/yjh/pipeline_work" \
#   --output_dir "$OUTPUT_DIR" \
#   --model "gemini-3-pro-preview"

# # 4. InSAR
python pipeline.py \
  --task_name "insar" \
  --gt_code "/home/yjh/insar_sandbox/insar_code.py" \
  --working_folder "/data/yjh/pipeline_work" \
  --output_dir "$OUTPUT_DIR" \
  --model "gemini-3-pro-preview"
