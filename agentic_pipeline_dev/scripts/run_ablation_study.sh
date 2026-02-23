#!/bin/bash
set -e

# Ensure we are in the project root (where run_task.py is)
if [ ! -f "run_task.py" ]; then
    echo "Error: run_task.py not found in current directory. Please run from project root."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Clean DB
echo "Cleaning database..."
rm -f persistent_skill_system/skills_new.db

# Generation Phase
echo "Starting Generation Phase (config/config_task.yaml)..."
# This step populates the database with skills from the training tasks
# We run this sequentially first because the ablation phase relies on this database
CUDA_VISIBLE_DEVICES=0 python run_task.py --config config/config_task.yaml --mode default > logs/generation.log 2>&1
if [ $? -ne 0 ]; then
    echo "Generation failed! Check logs/generation.log"
    exit 1
fi
echo "Generation Complete. Database populated."

# Ablation Phase
# Now we run the test set (config_task_2.yaml) with 4 different skill manager configurations
# The database is effectively frozen for these runs (except for Mode 1 which doesn't use it)
echo "Starting Ablation Phase (config/config_task_2.yaml)..."
echo "Running 4 modes in parallel..."

# Mode 1: Baseline (No Skills) -> GPU 0
CUDA_VISIBLE_DEVICES=0 python run_task.py --config config/config_task_2.yaml --mode none > logs/ablation_none.log 2>&1 &
PID1=$!
echo "Started Mode 1 (None) PID: $PID1"

# Mode 2: Instance Only -> GPU 1
CUDA_VISIBLE_DEVICES=1 python run_task.py --config config/config_task_2.yaml --mode instance > logs/ablation_instance.log 2>&1 &
PID2=$!
echo "Started Mode 2 (Instance) PID: $PID2"

# Mode 3: Experience Only -> GPU 2
CUDA_VISIBLE_DEVICES=2 python run_task.py --config config/config_task_2.yaml --mode experience > logs/ablation_experience.log 2>&1 &
PID3=$!
echo "Started Mode 3 (Experience) PID: $PID3"

# Mode 4: Instance + Experience -> GPU 3
CUDA_VISIBLE_DEVICES=3 python run_task.py --config config/config_task_2.yaml --mode instance_exp > logs/ablation_instance_exp.log 2>&1 &
PID4=$!
echo "Started Mode 4 (Instance+Exp) PID: $PID4"

# Wait for all background processes to finish
wait $PID1 $PID2 $PID3 $PID4

echo "Ablation Study Complete. Check logs/ for details."
