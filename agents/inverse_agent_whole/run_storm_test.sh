export CUDA_VISIBLE_DEVICES=7
OUTPUT_DIR="results_pipeline_gemini_storm_test"
mkdir -p /data/yjh/pipeline_work

# Clear previous cache for storm_analysis to ensure oracle is regenerated with new metrics
# rm -rf /data/yjh/benchmark_dataset/storm_analysis

python pipeline.py \
  --task_name "storm_analysis" \
  --gt_code "/data/yjh/storm-analysis-master_sandbox/storm_code.py" \
  --working_folder "/data/yjh/pipeline_work" \
  --output_dir "$OUTPUT_DIR" \
  --model "gemini-3-pro-preview"
