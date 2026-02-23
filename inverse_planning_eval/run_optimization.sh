#!/bin/bash
# tmux new -s docsgen
# tmux new -s yjh
# tmux new -s qwen
# tmux new -s claude

# tmux attach -t docsgen
# tmux attach -t yjh
# tmux attach -t qwen
# tmux attach -t claude

# Configuration
SCRIPT_PATH="/home/yjh/inverse_planning_eval/prompt_optimization.py"
BASE_DIR="/home/yjh/inverse_planning_eval"
ARTIFACTS_LOG="$BASE_DIR/optimization_artifacts.log"
EXECUTION_LOG="$BASE_DIR/optimization_execution.log"

BATCH_SIZE=4
EPOCHS=8

# Clear logs
echo "Clearing logs..."
> "$ARTIFACTS_LOG"
> "$EXECUTION_LOG"

# Models to tune (Comma separated)
MODELS="gemini-3-pro-preview,qwen3-max,deepseek-v3.2,gpt-5.2-thinking,glm-4.7"
OPTIMIZER_MODEL="claude-opus-4-5-20251101-thinking"

echo "Starting Prompt Optimization Loop (Joint Optimization)..."

python3 "$SCRIPT_PATH" \
    --models "$MODELS" \
    --optimizer_model "$OPTIMIZER_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS"
    
if [ $? -ne 0 ]; then
    echo "Error running optimization."
else
    echo "Optimization completed successfully."
fi
