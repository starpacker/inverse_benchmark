#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Sequential Benchmark Runner
# ═══════════════════════════════════════════════════════════════
# Runs all 26 tasks ONE AT A TIME to avoid resource contention
# that causes action_execution_server crashes in parallel mode.
#
# Usage:
#   bash launch_sequential_benchmark.sh                  # all 26 tasks
#   bash launch_sequential_benchmark.sh --skip-done      # skip tasks with existing results
#
# The script uses the unified config_task_2.yaml with all 26 tasks.
# Each task gets max 50 iterations with Claude 4.6 Opus.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${BENCH_RESULTS_DIR:-/data/openhands_results}"
LOG_DIR="${RESULTS_DIR}/logs"
CONDA_ENV="openhands"
TASK_CONFIG="${BENCH_TASK_CONFIG:-./config/tasks.yaml}"
MAX_ITERATIONS=50

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════"
echo " OpenHands Sequential Benchmark"
echo " $(date)"
echo "═══════════════════════════════════════════════════════"
echo " Config      : $TASK_CONFIG"
echo " Max Iter    : $MAX_ITERATIONS"
echo " Results Dir : $RESULTS_DIR"
echo " Log Dir     : $LOG_DIR"
echo "═══════════════════════════════════════════════════════"

# Clean up any leftover servers
pkill -f "action_execution_server" 2>/dev/null || true
sleep 2

# Run the benchmark sequentially
LOG_FILE="${LOG_DIR}/sequential_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

conda run -n "$CONDA_ENV" --no-capture-output \
    python "${SCRIPT_DIR}/run_openhands_benchmark_v2.py" \
        --task-config "$TASK_CONFIG" \
        --max-iterations "$MAX_ITERATIONS" \
        --gpu-id 0 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Done at $(date)"
echo " Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════"
