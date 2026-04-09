#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# OpenHands Benchmark — 8-GPU Parallel Launch
# ═══════════════════════════════════════════════════════════════════
# Launches 8 parallel OpenHands benchmark runners, one per GPU.
# Each reads its own config_task_2_gpu{i}.yaml (3-4 tasks each).
#
# Usage:
#   bash launch_parallel_benchmark.sh              # launch all 8
#   bash launch_parallel_benchmark.sh --dry-run    # show commands only
#   bash launch_parallel_benchmark.sh --monitor    # launch + monitor
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

OPENHANDS_DIR="${OPENHANDS_DIR:-/path/to/OpenHands}"
CONFIG_DIR="${BENCH_CONFIG_DIR:-./config}"
RESULTS_DIR="${BENCH_RESULTS_DIR:-/data/openhands_results}"
LOG_DIR="${RESULTS_DIR}/logs"
CONDA_ENV="openhands"
MAX_ITERATIONS=50
NUM_GPUS=8

DRY_RUN=false
MONITOR=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --monitor) MONITOR=true ;;
    esac
done

# ── Setup ──
mkdir -p "$LOG_DIR" "$RESULTS_DIR/trajectories"

echo "═══════════════════════════════════════════════════════════"
echo "  OpenHands 8-GPU Parallel Benchmark"
echo "═══════════════════════════════════════════════════════════"
echo "  Max iterations : $MAX_ITERATIONS"
echo "  Results dir    : $RESULTS_DIR"
echo "  Log dir        : $LOG_DIR"
echo "  Conda env      : $CONDA_ENV"
echo ""

# ── Kill any leftover processes ──
echo "Cleaning up old processes..."
pkill -f "action_execution_server" 2>/dev/null || true
pkill -f "run_openhands_benchmark_v2" 2>/dev/null || true
sleep 2

# ── Task distribution summary ──
echo ""
echo "Task distribution across GPUs:"
echo "─────────────────────────────────"
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    config_file="${CONFIG_DIR}/config_task_2_gpu${gpu_id}.yaml"
    if [[ -f "$config_file" ]]; then
        task_count=$(grep -c "^- " "$config_file" 2>/dev/null || echo "?")
        task_names=$(grep "name:" "$config_file" | awk '{print $2}' | tr '\n' ', ' | sed 's/,$//')
        echo "  GPU $gpu_id ($task_count tasks): $task_names"
    else
        echo "  GPU $gpu_id: CONFIG NOT FOUND: $config_file"
    fi
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Commands that would be executed:"
    echo ""
fi

# ── Launch workers ──
PIDS=()
START_TIME=$(date +%s)

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    config_file="${CONFIG_DIR}/config_task_2_gpu${gpu_id}.yaml"
    log_file="${LOG_DIR}/gpu${gpu_id}.log"

    if [[ ! -f "$config_file" ]]; then
        echo "⚠ Skipping GPU $gpu_id: $config_file not found"
        continue
    fi

    CMD="cd ${OPENHANDS_DIR} && conda run -n ${CONDA_ENV} --no-capture-output python run_openhands_benchmark_v2.py \
        --task-config ${config_file} \
        --gpu-id ${gpu_id} \
        --max-iterations ${MAX_ITERATIONS}"

    if $DRY_RUN; then
        echo "  [GPU $gpu_id] $CMD > $log_file 2>&1 &"
    else
        echo "🚀 Launching GPU $gpu_id → $log_file"
        nohup bash -c "$CMD" > "$log_file" 2>&1 &
        pid=$!
        PIDS+=("$gpu_id:$pid")
        echo "   PID: $pid"
        sleep 2  # stagger launches to avoid port collisions
    fi
done

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] No processes launched."
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All ${#PIDS[@]} workers launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Monitor commands:"
echo "    tail -f ${LOG_DIR}/gpu*.log              # all logs"
echo "    tail -f ${LOG_DIR}/gpu0.log              # specific GPU"
echo "    bash $0 --monitor                         # auto-monitor"
echo ""
echo "  Check status:"
echo "    ps aux | grep run_openhands_benchmark_v2  # running?"
echo ""
echo "  PIDs:"
for entry in "${PIDS[@]}"; do
    IFS=':' read -r gid pid <<< "$entry"
    echo "    GPU $gid → PID $pid"
done

# Save PIDs for later monitoring
PID_FILE="${RESULTS_DIR}/benchmark_pids.txt"
echo "# OpenHands Benchmark PIDs - $(date)" > "$PID_FILE"
echo "# START_TIME=$START_TIME" >> "$PID_FILE"
for entry in "${PIDS[@]}"; do
    echo "$entry" >> "$PID_FILE"
done
echo ""
echo "  PID file: $PID_FILE"

# ── Optional monitoring loop ──
if $MONITOR; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  MONITORING (Ctrl+C to stop monitoring, workers continue)"
    echo "═══════════════════════════════════════════════════════════"

    while true; do
        sleep 30
        echo ""
        echo "── Status at $(date '+%H:%M:%S') ──"

        all_done=true
        for entry in "${PIDS[@]}"; do
            IFS=':' read -r gid pid <<< "$entry"
            if kill -0 "$pid" 2>/dev/null; then
                all_done=false
                # Get last meaningful line from log
                status=$(tail -5 "${LOG_DIR}/gpu${gid}.log" 2>/dev/null | grep -E "Task|▶|✓|✗|⚠|SUMMARY" | tail -1 || echo "running...")
                echo "  GPU $gid [RUNNING] $status"
            else
                # Check exit code
                wait "$pid" 2>/dev/null
                ec=$?
                result_line=$(tail -3 "${LOG_DIR}/gpu${gid}.log" 2>/dev/null | head -1 || echo "finished")
                echo "  GPU $gid [DONE ec=$ec] $result_line"
            fi
        done

        if $all_done; then
            ELAPSED=$(( $(date +%s) - START_TIME ))
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "  ALL WORKERS FINISHED in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
            echo "═══════════════════════════════════════════════════════════"
            break
        fi
    done
fi
