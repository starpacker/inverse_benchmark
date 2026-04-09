import os
import yaml
import glob
import subprocess
import threading
import time

# Configuration
CONFIG_PATH = "config/config_previous.yaml"
MODELS = [
    {"name": "deepseek-v3.2", "gpu_id": "7"},
    {"name": "gpt-5.2-thinking", "gpu_id": "6"},
    {"name": "deepseek-r1-250528", "gpu_id": "5"},
    {"name": "grok-4", "gpu_id": "4"}
]

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def is_task_completed(sandbox_root, model_name, task_name):
    # Log path pattern: sandbox_root/model_name/task_name/logs/run_taskname_*.log
    # model_name in path might be sanitized if it contains slashes, but based on react_code_log_dec7.py:
    # task_sandbox_dir = os.path.join(sandbox_root, args.model_name, task_name)
    # It uses args.model_name directly. Linux allows / in paths, but here it would mean subdirectories.
    # Let's assume standard behavior: deepseek/deepseek-chat-v3.2 -> directory "deepseek" -> subdir "deepseek-chat-v3.2"
    
    # Check if directory exists
    log_dir = os.path.join(sandbox_root, model_name, task_name, "logs")
    
    if not os.path.exists(log_dir):
        return False
        
    # Check for any log file
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    return len(log_files) > 0

def run_model_tasks(model_config, tasks, sandbox_root):
    model_name = model_config["name"]
    gpu_id = model_config["gpu_id"]
    
    print(f"[{model_name}] Checking for missing tasks...")
    
    missing_tasks = []
    for task in tasks:
        task_name = task["name"]
        if not is_task_completed(sandbox_root, model_name, task_name):
            missing_tasks.append(task_name)
        else:
            # print(f"[{model_name}] Task {task_name} already completed. Skipping.")
            pass
            
    if not missing_tasks:
        print(f"[{model_name}] All tasks completed! 🎉")
        return

    print(f"[{model_name}] Found {len(missing_tasks)} missing tasks: {missing_tasks}")
    
    for task_name in missing_tasks:
        print(f"[{model_name}] Starting task: {task_name} on GPU {gpu_id}")
        
        cmd = [
            "python", "react_code_log_dec7.py",
            "--model_name", model_name,
            "--gpu_id", gpu_id,
            "--task", task_name,
            "--config_path", CONFIG_PATH
        ]
        
        try:
            # Run command synchronously for this model (sequential tasks)
            # We capture output to avoid terminal clutter, or redirect to a file?
            # Let's just inherit stdout/stderr but maybe prefix it? 
            # For simplicity in this script, we assume the tool logs to files anyway.
            
            # Using Popen to wait
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"[{model_name}] Task {task_name} finished successfully.")
            else:
                print(f"[{model_name}] Task {task_name} failed with code {process.returncode}.")
                if stderr:
                    print(f"[{model_name}] Error output: {stderr.decode()}")
                    
        except Exception as e:
            print(f"[{model_name}] Error running task {task_name}: {e}")

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    config = load_config(CONFIG_PATH)
    sandbox_root = config["global"]["sandbox_root"]
    tasks = config["tasks"]
    
    print(f"Loaded {len(tasks)} tasks from config.")
    print(f"Sandbox Root: {sandbox_root}")
    
    threads = []
    
    # Launch a thread for each model to run in parallel
    for model_config in MODELS:
        t = threading.Thread(target=run_model_tasks, args=(model_config, tasks, sandbox_root))
        t.start()
        threads.append(t)
        
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    print("\nAll models have finished processing their tasks.")

if __name__ == "__main__":
    main()
