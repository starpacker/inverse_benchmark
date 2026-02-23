import sqlite3
import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, Any

# Configuration
DB_PATH = "/home/yjh/agentic_pipeline/persistent_skill_system/skills_new.db"
EXPORT_ROOT = f"/home/yjh/knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_directories():
    os.makedirs(EXPORT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(EXPORT_ROOT, "core_knowledge"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_ROOT, "experiences"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_ROOT, "instances"), exist_ok=True)
    print(f"📁 Created export directory: {EXPORT_ROOT}")

def format_json(data: Any) -> str:
    """Pretty print JSON for markdown."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return data
    return json.dumps(data, indent=2, ensure_ascii=False)

def export_knowledge_items(cursor):
    print("🧠 Exporting Knowledge Items...")
    cursor.execute("SELECT * FROM knowledge_items ORDER BY credit_score DESC")
    items = cursor.fetchall()
    
    files = {
        "core": open(os.path.join(EXPORT_ROOT, "core_knowledge", "core_knowledge.md"), "w", encoding="utf-8"),
        "experience": open(os.path.join(EXPORT_ROOT, "experiences", "experiences.md"), "w", encoding="utf-8"),
        "instance": open(os.path.join(EXPORT_ROOT, "instances", "instances.md"), "w", encoding="utf-8")
    }
    
    # Headers
    for f in files.values():
        f.write("# Knowledge Export\n\n")

    counts = {"core": 0, "experience": 0, "instance": 0}

    for item in items:
        # Convert sqlite3.Row to dict to support .get() and modification
        item = dict(item)
        
        k_type = item["type"]
        if k_type not in files:
            continue
            
        f = files[k_type]
        counts[k_type] += 1
        
        content = json.loads(item["content"])
        
        f.write(f"## {item['name']}\n")
        f.write(f"- **ID**: `{item['id']}`\n")
        f.write(f"- **Score**: {item['credit_score']:.2f} | **Usage**: {item['usage_count']}\n")
        f.write(f"- **Scope**: {item['agent_scope']} | **Tags**: {item['tags']}\n")
        f.write(f"- **Artifact Type**: {item.get('artifact_type', 'unknown')}\n")
        f.write(f"- **Source Tasks**: {item['source_trajectories']}\n\n")
        
        if k_type == "experience":
            f.write(f"### Condition\n{content.get('condition', 'N/A')}\n\n")
            f.write(f"### Action\n{content.get('action', 'N/A')}\n\n")
            f.write(f"### Rationale\n{content.get('rationale', 'N/A')}\n\n")
            
        elif k_type == "core":
            f.write(f"### Principle\n{content.get('principle', 'N/A')}\n\n")
            if 'checklist' in content:
                f.write("### Checklist\n")
                for check in content['checklist']:
                    f.write(f"- [ ] {check}\n")
                f.write("\n")
                
        elif k_type == "instance":
            # Determine artifact type from DB column or content structure
            artifact_type = item.get('artifact_type', 'unknown')
            
            # Extract description if content is a dict
            if isinstance(content, dict):
                desc = content.get('description', '')
                code = content.get('code', '')
                plan = content.get('plan', '')
                skeleton = content.get('skeleton', '')
            else:
                # Content is a raw string (plan text, code text, etc.)
                desc = ""
                code = ""
                plan = ""
                skeleton = ""
            
            if desc:
                f.write(f"**Description**: {desc}\n\n")
            
            # Route based on artifact_type (from DB column, not content keys)
            if artifact_type == 'plan':
                # Plan content is typically stored as a raw string in content field
                plan_text = content if isinstance(content, str) else (plan or str(content))
                f.write("### Plan\n")
                f.write(str(plan_text))  # Full content, no truncation
                f.write("\n\n")
                
            elif artifact_type == 'skeleton':
                # Skeleton is Python code
                skel_text = content if isinstance(content, str) else (skeleton or str(content))
                f.write("### Skeleton Code\n```python\n")
                f.write(str(skel_text))  # Full content, no truncation
                f.write("\n```\n\n")
                
            elif artifact_type == 'code':
                # Solution code is Python
                code_text = content if isinstance(content, str) else (code or str(content))
                f.write("### Solution Code\n```python\n")
                f.write(str(code_text))  # Full content, no truncation
                f.write("\n```\n\n")
                
            elif artifact_type == 'feedback':
                # Feedback is a JSON dict
                f.write("### Feedback / Evaluation Logic\n```json\n")
                f.write(format_json(content))  # Full content, no truncation
                f.write("\n```\n\n")
                
            else:
                # Unknown artifact type - output as JSON
                f.write(f"### Content (artifact_type: {artifact_type})\n```json\n")
                f.write(format_json(content))  # Full content, no truncation
                f.write("\n```\n\n")
        
        f.write("---\n\n")

    for f in files.values():
        f.close()
        
    print(f"  - Core: {counts['core']}")
    print(f"  - Experiences: {counts['experience']}")
    print(f"  - Instances: {counts['instance']}")
    return counts

def generate_summary(k_counts):
    with open(os.path.join(EXPORT_ROOT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Knowledge Base Export Summary\n\n")
        f.write(f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Statistics\n")
        f.write("| Category | Count |\n")
        f.write("|----------|-------|\n")
        f.write(f"| **Knowledge Items** | {sum(k_counts.values())} |\n")
        f.write(f"| - Core Principles | {k_counts['core']} |\n")
        f.write(f"| - Experiences | {k_counts['experience']} |\n")
        f.write(f"| - Instances | {k_counts['instance']} |\n")
        
        f.write("\n## 📁 Directory Structure\n")
        f.write("- `core_knowledge/`: Fundamental principles extracted from tasks.\n")
        f.write("- `experiences/`: Troubleshooting patterns and strategies.\n")
        f.write("- `instances/`: Few-shot examples (plans/skeletons/code/feedback).\n")

def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    setup_directories()
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    k_counts = export_knowledge_items(cursor)
    
    conn.close()
    
    generate_summary(k_counts)
    print("\n✅ Export Complete!")

if __name__ == "__main__":
    main()
