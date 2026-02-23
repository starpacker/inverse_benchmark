# 创建工作目录
mkdir -p /data/yjh/ablation_work

# 1. Spectral CT
python pipeline_ablation.py \
  --task_name "spectral_ct" \
  --gt_code "/data/yjh/spectral_ct_examples-master_sandbox/spectral_ct_code.py" \
  --working_folder "/data/yjh/ablation_work" \
  --output_dir "results_ablation_gemini" \
  --model "gemini-3-pro-preview"

# 2. STORM Analysis
python pipeline_ablation.py \
  --task_name "storm_analysis" \
  --gt_code "/data/yjh/storm-analysis-master_sandbox/storm_code.py" \
  --working_folder "/data/yjh/ablation_work" \
  --output_dir "results_ablation_gemini" \
  --model "gemini-3-pro-preview"

# 3. BPM (Beam Propagation Method)
python pipeline_ablation.py \
  --task_name "bpm" \
  --gt_code "/home/yjh/bpm_sandbox/bpm_code.py" \
  --working_folder "/data/yjh/ablation_work" \
  --output_dir "results_ablation_gemini" \
  --model "gemini-3-pro-preview"

# 4. InSAR
python pipeline_ablation.py \
  --task_name "insar" \
  --gt_code "/home/yjh/insar_sandbox/insar_code.py" \
  --working_folder "/data/yjh/ablation_work" \
  --output_dir "results_ablation_gemini" \
  --model "gemini-3-pro-preview"