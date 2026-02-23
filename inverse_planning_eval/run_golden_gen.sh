#!/bin/bash
# run_golden_gen.sh
# Wrapper script to generate Golden Plans

echo "Starting Golden Plan Generation..."
python3 /home/yjh/inverse_planning_eval/generate_golden_plans.py

if [ $? -eq 0 ]; then
    echo "Golden Plan Generation Completed Successfully."
    echo "Results saved to /data/yjh/golden_plans"
else
    echo "Golden Plan Generation Failed."
    exit 1
fi
