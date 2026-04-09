
import os
import sys
import json
import sqlite3
import yaml
import argparse
import time
import numpy as np
from typing import List, Dict, Any

# Add path for imports
sys.path.append("/home/yjh")
sys.path.append("/home/yjh/agentic_pipeline")

from agentic_pipeline.persistent_skill_system.manager import SkillManager

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_skill_manager():
    base_dir = "/home/yjh/agentic_pipeline"
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")
    llm_config = load_config(llm_config_path)
    model_key = "cds/Claude-4.6-opus"
    
    # Check if model key exists, fallback if needed or raise error
    if model_key not in llm_config['models']:
         # Try to find another available model
         keys = list(llm_config['models'].keys())
         if keys:
             model_key = keys[0]
         else:
             raise ValueError("No models found in config")
             
    model_conf = llm_config['models'][model_key]
    
    from openai import OpenAI
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )
    
    skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    return SkillManager(db_path=skill_db_path, client=client, model_name=model_key)

def list_knowledge(manager, k_type=None):
    items = manager.storage.get_knowledge_items(k_type=k_type)
    
    # Sort by update time (newest first)
    items.sort(key=lambda x: x['update_time'], reverse=True)
    
    type_label = k_type.upper() if k_type else "ALL"
    print(f"\n📚 Found {len(items)} {type_label} Knowledge Items:")
    print("-" * 100)
    print(f"{'ID':<4} | {'Type':<10} | {'Scope':<10} | {'Name':<40} | {'Status'}")
    print("-" * 100)
    
    for i, item in enumerate(items):
        name = item['name']
        if len(name) > 38:
            name = name[:35] + "..."
        
        k_type_str = item['type']
        scope = item.get('agent_scope', 'General') or 'General'
        status = item.get('status', 'active')
        
        print(f"{i+1:<4} | {k_type_str:<10} | {scope:<10} | {name:<40} | {status}")
        
    print("-" * 100)
    return items

def delete_knowledge(manager, item_id):
    conn = sqlite3.connect(manager.storage.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    print(f"✅ Deleted item {item_id}")

def edit_knowledge(manager, item):
    print(f"\nEditing: {item['name']} (ID: {item['id']})")
    
    new_name = input(f"Name [{item['name']}]: ").strip()
    if not new_name: new_name = item['name']
    
    content = item['content']
    
    # Edit specific fields based on type
    if item['type'] == 'experience':
        print(f"\nCurrent Condition: {content.get('condition', '')}")
        new_cond = input("New Condition (Enter to keep): ").strip()
        if new_cond: content['condition'] = new_cond
        
        print(f"\nCurrent Action: {content.get('action', '')}")
        new_action = input("New Action (Enter to keep): ").strip()
        if new_action: content['action'] = new_action
        
        print(f"\nCurrent Rationale: {content.get('rationale', '')}")
        new_rat = input("New Rationale (Enter to keep): ").strip()
        if new_rat: content['rationale'] = new_rat
        
    elif item['type'] == 'core':
        print(f"\nCurrent Principle: {content.get('principle', '')}")
        new_principle = input("New Principle (Enter to keep): ").strip()
        if new_principle: content['principle'] = new_principle
        
    # Re-embed if content changed (simplified logic: re-embed if name or content changed)
    # For robust re-embedding, we should reconstruct the text representation based on type
    emb_text = ""
    if item['type'] == 'experience':
        emb_text = f"{new_name} {content.get('condition', '')} {content.get('action', '')}"
    elif item['type'] == 'core':
        emb_text = f"{new_name} {content.get('principle', '')}"
    else:
        emb_text = f"{new_name} {str(content)}"
        
    embedding = manager.get_embedding(emb_text)
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    conn = sqlite3.connect(manager.storage.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE knowledge_items 
        SET name = ?, content = ?, embedding = ?, update_time = ?
        WHERE id = ?
    ''', (new_name, json.dumps(content), embedding_bytes, int(time.time()), item['id']))
    conn.commit()
    conn.close()
    print("✅ Updated successfully.")

def interactive_mode():
    print("Initializing Knowledge Manager...")
    try:
        manager = get_skill_manager()
    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return

    current_type = None # Default to All
    
    while True:
        items = list_knowledge(manager, k_type=current_type)
        print("\nOptions:")
        print("  [Filter] t <type> : Filter by type (core, experience, instance, all)")
        print("  [Action] d <index>: Delete item")
        print("  [Action] e <index>: Edit item")
        print("  q                 : Quit")
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == 'q':
            break
            
        try:
            parts = choice.split()
            if not parts: continue
            
            cmd = parts[0]
            
            if cmd == 't':
                if len(parts) < 2:
                    print("❌ Specify type: core, experience, instance, all")
                    continue
                new_type = parts[1]
                if new_type in ['core', 'experience', 'instance']:
                    current_type = new_type
                elif new_type == 'all':
                    current_type = None
                else:
                    print("❌ Invalid type.")
                continue
                
            if len(parts) < 2:
                print("❌ Invalid command format.")
                continue
                
            idx = int(parts[1]) - 1
            
            if idx < 0 or idx >= len(items):
                print("❌ Invalid index.")
                continue
                
            target = items[idx]
            
            if cmd == 'd':
                confirm = input(f"Are you sure you want to DELETE '{target['name']}'? (y/n): ")
                if confirm.lower() == 'y':
                    delete_knowledge(manager, target['id'])
            elif cmd == 'e':
                edit_knowledge(manager, target)
            else:
                print("❌ Unknown command.")
        except ValueError:
            print("❌ Invalid format.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_mode()
