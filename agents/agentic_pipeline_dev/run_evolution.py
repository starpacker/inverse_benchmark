
import os
import sys
import yaml
import time
import argparse
from openai import OpenAI

# Add path for imports
sys.path.append("/home/yjh")
sys.path.append("/home/yjh/agentic_pipeline")

from agentic_pipeline.persistent_skill_system.manager import SkillManager
from agentic_pipeline.persistent_skill_system.evolution_manager import EvolutionManager

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_evolution_once():
    print(">>> Starting Single Evolution Cycle")
    
    # 1. Setup Configuration
    base_dir = "/home/yjh/agentic_pipeline"
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")
    llm_config = load_config(llm_config_path)
    
    model_key = "cds/Claude-4.6-opus"
    model_conf = llm_config['models'][model_key]
    
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )
    
    skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    
    # 2. Initialize Managers
    skill_manager = SkillManager(
        db_path=skill_db_path,
        client=client,
        model_name=model_key
    )
    
    evolution_manager = EvolutionManager(skill_manager)
    
    # 3. Continuous Loop (Wait for new data)
    iteration = 0
    last_processed_count = 0
    
    while True:
        # Check if new experiences have been added
        current_experiences = skill_manager.storage.get_knowledge_items(k_type='experience')
        current_count = len(current_experiences)
        
        if current_count > last_processed_count:
            iteration += 1
            print(f"\n[{time.strftime('%H:%M:%S')}] New data detected ({current_count} > {last_processed_count}). Starting Evolution Cycle #{iteration}...")
            
            try:
                evolution_manager.run_evolution_loop()
                last_processed_count = current_count # Update watermark
            except Exception as e:
                print(f"❌ Error in evolution loop: {e}")
                import traceback
                traceback.print_exc()
                
            print(f"[{time.strftime('%H:%M:%S')}] Cycle #{iteration} Complete.")
        else:
            # print(f"[{time.strftime('%H:%M:%S')}] No new experiences. Waiting...")
            pass
            
        time.sleep(30) # Check every 30 seconds

def run_evolution_once():
    print(">>> Starting Single Evolution Cycle")
    
    # 1. Setup Configuration
    base_dir = "/home/yjh/agentic_pipeline"
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")
    llm_config = load_config(llm_config_path)
    
    model_key = "cds/Claude-4.6-opus"
    model_conf = llm_config['models'][model_key]
    
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )
    
    skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    
    # 2. Initialize Managers
    skill_manager = SkillManager(
        db_path=skill_db_path,
        client=client,
        model_name=model_key
    )
    
    evolution_manager = EvolutionManager(skill_manager)
    
    # 3. Run Once
    print(f"[{time.strftime('%H:%M:%S')}] Starting Evolution Cycle...")
    
    try:
        evolution_manager.run_evolution_loop()
    except Exception as e:
        print(f"❌ Error in evolution loop: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"[{time.strftime('%H:%M:%S')}] Evolution Cycle Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true", help="Run in continuous daemon mode")
    args = parser.parse_args()
    
    if args.daemon:
        run_evolution_daemon()
    else:
        run_evolution_once()
