#!/bin/bash
# run_eval.sh
# Wrapper script to run the Evaluation Pipeline

# Output file path
OUTPUT_FILE="/home/yjh/inverse_planning_eval/eval_results.json"

echo "Starting Evaluation Pipeline..."
python3 /home/yjh/inverse_planning_eval/run_eval.py --output_file "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Evaluation Completed Successfully."
    echo "Results saved to $OUTPUT_FILE"
else
    echo "Evaluation Failed."
    exit 1
fi
