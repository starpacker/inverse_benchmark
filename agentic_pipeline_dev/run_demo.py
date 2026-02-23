
import os
import sys
import yaml
import time
import json
import traceback
from pathlib import Path
from openai import OpenAI

# Add path for imports
sys.path.append("/home/yjh")
sys.path.append("/home/yjh/agentic_pipeline")
from agentic_pipeline.main_flow_test_only import InverseProblemWorkflowTestOnly
from agentic_pipeline.persistent_skill_system.manager import SkillManager

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_demo():
    print(">>> Starting Demo: Run Task + Distill Knowledge")
    
    # 1. Setup Configuration
    base_dir = "/home/yjh/agentic_pipeline"
    task_config_path = os.path.join(base_dir, "config", "config_task_2.yaml")
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")
    
    task_config = load_config(task_config_path)
    llm_config = load_config(llm_config_path)
    
    # Model Selection
    model_key = "cds/Claude-4.6-opus"
    model_conf = llm_config['models'][model_key]
    
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )
    
    # 2. Initialize Skill Manager
    skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    skill_manager = SkillManager(
        db_path=skill_db_path,
        client=client,
        model_name=model_key
    )
    print(f"✓ Skill Manager initialized at {skill_db_path}")

    # 3. Select Task (tomopy-master)
    target_task_name = "tomopy-master"
    task_info = next((t for t in task_config['tasks'] if t['name'] == target_task_name), None)
    if not task_info:
        print(f"❌ Task {target_task_name} not found in config.")
        return

    print(f"▶ Selected Task: {task_info['name']}")
    
    # 4. Run Workflow
    # Load description from file
    desc_path = os.path.join("/data/yjh/task_descriptions", f"{task_info['name']}_description.md")
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            task_desc = f.read()
        print(f"✓ Loaded task description from {desc_path}")
    else:
        print(f"⚠ Description not found at {desc_path}, using default.")
        task_desc = f"Reconstruct tomography data. Task: {task_info['name']}"

    # Note: We use TestOnly workflow, which saves trajectory but skips distillation during run
    workflow = InverseProblemWorkflowTestOnly(
        task_name=task_info['name'],
        task_desc=task_desc,
        gt_code_path=task_info['gt_code_path'],
        python_path=task_info.get('python_path', 'python'),
        working_dir=task_info['working_folder'],
        client=client,
        model_name=model_key,
        root_output_dir="/data/yjh/end_sandbox",
        skill_manager=skill_manager
    )
    
    # Force max_retries to small number for demo speed if needed, but let's keep default
    workflow.max_retries = 3 
    
    print("\n>>> Executing Workflow...")
    success = workflow.run()
    
    print(f"\n>>> Workflow Finished. Success: {success}")
    
    # 5. Retrieve and Distill
    # Even if failed, we might want to distill diagnosis knowledge
    print("\n>>> retrieving stored trajectory for distillation...")
    
    # Get latest trajectory for this task
    # We use the storage directly to find it
    trajs = skill_manager.storage.get_trajectories(limit=5, load_full=True)
    target_traj = None
    for t in trajs:
        if t['task_name'] == task_info['name']:
            target_traj = t
            break
            
    if target_traj:
        print(f"✓ Found trajectory: {target_traj.get('exp_id')} (Outcome: {target_traj.get('outcome')})")
        print(f"  Retrieval Key: {target_traj.get('retrieval_key')}")
        
        # Manually Trigger Distillation
        print("\n>>> 🧠 Triggering Knowledge Distillation...")
        skill_manager.distill_and_store(target_traj)
        
        print("\n>>> ✨ Distillation Complete.")
        
        # 6. Show Results
        print("\n" + "="*50)
        print("🔍 DEMONSTRATION RESULTS")
        print("="*50)
        
        # A. Trajectory File
        if 'file_path' in target_traj:
            print(f"\n[Trajectory File] {target_traj['file_path']}")
            if os.path.exists(target_traj['file_path']):
                size = os.path.getsize(target_traj['file_path'])
                print(f"  Size: {size} bytes")
                # Show snippet of steps
                steps = target_traj.get('steps', [])
                print(f"  Total Steps: {len(steps)}")
                for s in steps[:3]:
                    print(f"    - Step {s['step_id']} ({s['role']}): {s.get('retrieval_key')}")
        
        # B. Knowledge Items
        print("\n[Extracted Knowledge in DB]")
        # Retrieve recent items
        conn = sqlite3.connect(skill_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, type, agent_scope FROM knowledge_items ORDER BY create_time DESC LIMIT 5")
        items = cursor.fetchall()
        conn.close()
        
        if items:
            for item in items:
                print(f"  - [{item[2].upper()}] {item[1]} (Scope: {item[3]})")
        else:
            print("  (No new knowledge items found)")
            
    else:
        print("❌ Could not find the stored trajectory.")

if __name__ == "__main__":
    run_demo()
