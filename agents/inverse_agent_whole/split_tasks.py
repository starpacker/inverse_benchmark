import yaml
import os
import math

def split_tasks():
    config_path = "/home/yjh/inverse_agent_whole/config/gt_code_index.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tasks = config.get('tasks', [])
    
    # Find the index of the last completed task 'lensless_dl'
    start_index = 0
    found = False
    for i, task in enumerate(tasks):
        if task.get('name') == 'lensless_dl':
            start_index = i + 1
            found = True
            break
    
    if not found:
        print("Warning: 'lensless_dl' not found in tasks. Starting from beginning or check names.")
        # If not found, maybe just start from beginning? Or fail?
        # User said "sim...lensless_dl are completed". So it must be there.
        # Let's print names to debug if it fails.
        pass

    remaining_tasks = tasks[start_index:]
    print(f"Total tasks: {len(tasks)}")
    print(f"Remaining tasks: {len(remaining_tasks)}")
    
    num_splits = 8
    chunk_size = math.ceil(len(remaining_tasks) / num_splits)
    
    for i in range(num_splits):
        chunk = remaining_tasks[i*chunk_size : (i+1)*chunk_size]
        
        split_config = {
            'global': config.get('global', {}),
            'tasks': chunk
        }
        
        output_path = f"/home/yjh/inverse_agent_whole/config/tasks_split_{i}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(split_config, f, default_flow_style=False)
        print(f"Created {output_path} with {len(chunk)} tasks.")

if __name__ == "__main__":
    split_tasks()
