import os
import yaml
import multiprocessing
import time
import sys
import traceback

# Ensure we can import from local package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import InverseAgentPipeline
from config import settings

# Single model mode as requested
MODEL_NAME = "gemini-3-pro-preview"

# If you want to run other models later, you can change this or add command line args
MODELS = [
    "gemini-3-pro-preview",
    "qwen3-max",
    "deepseek-v3.2",
    "gpt-5.2-thinking",
    "claude-opus-4-5-20251101-thinking",
    "kimi-k2-250905",
    "glm-4.7"
]

def run_model_tasks(model_name, gpu_id, tasks, output_root):
    print(f"[Worker] Starting worker for {model_name} on GPU {gpu_id} with {len(tasks)} tasks.")
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    for task in tasks:
        task_name = task.get('name')
        gt_code = task.get('gt_code_path')
        working_folder = task.get('working_folder')
        
        if not task_name or not gt_code or not working_folder:
            print(f"[{model_name}] Skipping invalid task config: {task}")
            continue

        print(f"[{model_name}] Starting task: {task_name} on GPU {gpu_id}")
        try:
            # Create pipeline instance per task to ensure clean state
            # Output will be in output_root/model_name/task_name
            task_out_dir = os.path.join(model_output_dir, task_name)
            
            # Initialize pipeline
            pipeline = InverseAgentPipeline(task_out_dir, model_name=model_name)
            
            # Run pipeline
            pipeline.run(task_name, gt_code, working_folder)
            
            print(f"[{model_name}] Finished task: {task_name}")
            
        except Exception as e:
            print(f"[{model_name}] Failed task {task_name}: {e}")
            traceback.print_exc()

def main():
    # Config paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = "/data/yjh/results_suite_final"
    
    processes = []
    
    # Launch 8 processes, one per GPU, each with a split config
    for i in range(8):
        gpu_id = i
        config_path = os.path.join(base_dir, f"config/tasks_split_{i}.yaml")
        
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            continue
            
        print(f"Loading tasks from {config_path} for GPU {gpu_id}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        tasks = config.get('tasks', [])
        print(f"Found {len(tasks)} tasks for GPU {gpu_id}.")
        
        if not tasks:
            print(f"No tasks for GPU {gpu_id}, skipping.")
            continue
            
        # Create and start process
        p = multiprocessing.Process(
            target=run_model_tasks, 
            args=(MODEL_NAME, gpu_id, tasks, output_root)
        )
        p.start()
        processes.append(p)
        print(f"Launched process for {MODEL_NAME} on GPU {gpu_id} (PID: {p.pid})")
        
    # Wait for all processes
    for p in processes:
        p.join()
        
    print("All distributed tasks completed.")

if __name__ == "__main__":
    main()
