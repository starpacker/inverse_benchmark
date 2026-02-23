
import os
import sys
import yaml
import time
import json
import sqlite3
import shutil
import numpy as np
from openai import OpenAI

# Add path for imports
sys.path.append("/home/yjh")
sys.path.append("/home/yjh/agentic_pipeline")

from agentic_pipeline.persistent_skill_system.manager import SkillManager
from agentic_pipeline.persistent_skill_system.evolution_manager import EvolutionManager

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_test_env():
    print(">>> Setting up Test Environment...")
    base_dir = "/home/yjh/agentic_pipeline"
    original_db = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    test_db = os.path.join(base_dir, "persistent_skill_system", "skills_test_evolution.db")
    
    # Copy DB to avoid messing up production
    if os.path.exists(test_db):
        os.remove(test_db)
    shutil.copy2(original_db, test_db)
    print(f"  ✓ Copied DB to {test_db}")
    
    return test_db

def create_mock_experience(manager, name, condition, action, rationale, principle_group="Group A"):
    """Helper to inject controlled experiences for testing clustering."""
    content = {
        "condition": condition,
        "action": action,
        "rationale": rationale
    }
    # Embed based on content to ensure clustering
    emb_text = f"{name} {condition} {action} {principle_group}" # Add group to text to force similarity
    embedding = manager.get_embedding(emb_text)
    
    item_data = {
        "name": name,
        "type": "experience",
        "content": content,
        "embedding": embedding,
        "tags": ["test_mock"],
        "source_trajectories": [],
        "agent_scope": "General",
        "status": "active"
    }
    manager.storage.add_knowledge_item(item_data)
    return item_data

def test_evolution_logic():
    print("\n" + "="*50)
    print("🧪 TESTING EVOLUTIONARY LOGIC (Knowledge Induction & Merge)")
    print("="*50)
    
    # 1. Setup
    test_db_path = setup_test_env()
    
    llm_config_path = "/home/yjh/agentic_pipeline/config/config_llm.yaml"
    llm_config = load_config(llm_config_path)
    model_key = "cds/Claude-4.6-opus"
    model_conf = llm_config['models'][model_key]
    
    client = OpenAI(api_key=model_conf['api_key'], base_url=model_conf['base_url'])
    
    skill_manager = SkillManager(db_path=test_db_path, client=client, model_name=model_key)
    evolution_manager = EvolutionManager(skill_manager)
    
    # 2. Inject Mock Experiences (Scenario: Shape Mismatch in Tomography)
    print("\n>>> [Step 1] Injecting Mock Experiences (Clustering Test)...")
    
    # Exp 1: Tomopy reshape
    create_mock_experience(skill_manager, 
        "Tomopy Shape Error Fix", 
        "When tomopy.recon throws dimension mismatch error on 2D input", 
        "Reshape input to (1, angles, width) before passing to recon function",
        "Tomopy expects 3D input even for single slice", 
        principle_group="3D_Input_Requirement"
    )
    
    # Exp 2: Astra reshape
    create_mock_experience(skill_manager, 
        "Astra Projection Dimension Fix", 
        "When astra.create_projector fails with shape error", 
        "Ensure detector width matches the second dimension of the sinogram 3D array",
        "Physics libraries require explicit 3D volume dimensions (slices, angles, width)", 
        principle_group="3D_Input_Requirement"
    )
    
    # Exp 3: Unrelated (Noise)
    create_mock_experience(skill_manager, 
        "Matplotlib Display Fix", 
        "When plt.imshow shows blank image", 
        "Check value range and use vmin/vmax",
        "Data might be float with very small values",
        principle_group="Visualization"
    )
    
    print("  ✓ Injected 3 mock experiences.")
    
    # 3. Run Evolution Cycle (Induction)
    print("\n>>> [Step 2] Running Evolution Cycle (Induction)...")
    evolution_manager.run_evolution_loop()
    
    # 4. Verify Core Knowledge Creation
    print("\n>>> [Step 3] Verifying Core Knowledge Creation...")
    ck_items = skill_manager.storage.get_knowledge_items(k_type='core')
    
    new_ck = None
    for item in ck_items:
        # Check for keywords related to 3D input requirement
        if any(kw in item['name'] for kw in ["3D", "Dimension", "Shape", "Input"]) or \
           any(kw in item['content']['principle'] for kw in ["3D", "Dimension", "Shape", "Input"]):
            new_ck = item
            print(f"  ✅ FOUND Generated CK: {item['name']}")
            print(f"     ID: {item['id']}")
            print(f"     Principle: {item['content']['principle'][:100]}...")
            print(f"     Status: {item['status']}")
            # source_experience_ids is already a list due to storage.py loading it
            source_ids = item['source_experience_ids']
            if isinstance(source_ids, str):
                source_ids = json.loads(source_ids)
                
            print(f"     Source Exp IDs: {len(source_ids)}")
            break
            
    if not new_ck:
        print("  ❌ Failed to induce Core Knowledge for 3D Input Requirement.")
        # Debug: Print what WAS generated
        print("  Available Core Knowledge:")
        for item in ck_items:
             print(f"  - {item['name']}")
        return

    # 5. Test Knowledge Merging (Refinement)
    print("\n>>> [Step 4] Testing Knowledge Merging (Refinement)...")
    
    # Inject a new experience that overlaps but adds detail
    create_mock_experience(skill_manager, 
        "Skimage Radon 3D Requirement", 
        "When radon transform output has wrong shape", 
        "Input image must be padded to square if using 3D wrappers",
        "Geometric consistency for 3D reconstruction libraries",
        principle_group="3D_Input_Requirement"
    )
    
    print("  ✓ Injected 1 new overlapping experience.")
    print("  Running Evolution Cycle again...")
    
    # Capture old version
    old_version = new_ck['version']
    old_id = new_ck['id']
    
    evolution_manager.run_evolution_loop()
    
    # Check if CK was updated
    updated_ck = skill_manager.storage.get_knowledge_by_ids([old_id])[0]
    
    print(f"  Checking CK {old_id}...")
    print(f"     Old Version: {old_version}")
    print(f"     New Version: {updated_ck['version']}")
    
    if updated_ck['version'] > old_version:
        print("  ✅ SUCCESS: Core Knowledge was merged/refined (Version incremented).")
        sources = updated_ck['source_experience_ids']
        if isinstance(sources, str): sources = json.loads(sources)
        print(f"     New Source Count: {len(sources)} (Expected > 2)")
    else:
        print("  ⚠️ WARNING: Core Knowledge version did not increment. (Maybe Critic decided 'Discard' or 'Create New'?)")
        
        # Check if a duplicate was created instead
        all_cks = skill_manager.storage.get_knowledge_items(k_type='core')
        if len(all_cks) > len(ck_items):
            print("  ⚠️ A NEW Core Knowledge was created instead of merging.")
            
    # 6. Clean up
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("\nTest Complete.")

if __name__ == "__main__":
    test_evolution_logic()
