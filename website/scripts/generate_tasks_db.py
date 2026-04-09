#!/usr/bin/env python3
"""Generate tasks_db.json from Report.md, domain_mapping.json, and task descriptions."""
import json
import os
import re
import glob

# Load domain mapping
with open("/home/yjh/docs/plan/agent-imaging-website/domain_mapping.json", "r") as f:
    domain_data = json.load(f)

domain_mapping = domain_data["domain_mapping"]
domain_labels = domain_data["domain_labels"]

# Parse Report.md for metrics and asset info
report_path = "/data/yjh/website_assets/Report.md"
with open(report_path, "r") as f:
    report_text = f.read()

# Parse each task section
tasks_db = {}
desc_dir = "/data/yjh/task_descriptions"
assets_dir = "/data/yjh/website_assets"

for task_id_str, dm in domain_mapping.items():
    task_id = int(task_id_str)
    task_id_padded = f"{task_id:02d}"
    
    # Find the task section in Report.md
    pattern = rf"## Task {task_id}: (.+?)(?=\n## Task \d|\n---\n## Task \d|\Z)"
    match = re.search(pattern, report_text, re.DOTALL)
    
    task_name = ""
    psnr = None
    ssim = None
    eval_type = ""
    
    if match:
        section = match.group(0)
        # Extract task name from header
        name_match = re.search(rf"## Task {task_id}: (.+)", section)
        if name_match:
            task_name = name_match.group(1).strip()
        
        # Extract PSNR
        psnr_match = re.search(r"\*\*PSNR\*\*: ([\d.]+) dB", section)
        if psnr_match:
            psnr = float(psnr_match.group(1))
        
        # Extract SSIM
        ssim_match = re.search(r"\*\*SSIM\*\*: ([\d.]+)", section)
        if ssim_match:
            ssim = float(ssim_match.group(1))
        
        # Extract eval type
        eval_match = re.search(r"\*\*评估类型\*\*: (.+)", section)
        if eval_match:
            eval_type = eval_match.group(1).strip()
    
    # Find task description file
    description = ""
    # Try to find matching description file
    possible_names = [
        task_name,
        task_name.replace("-", "_"),
        task_name.lower(),
    ]
    
    desc_files = glob.glob(os.path.join(desc_dir, "*_description.md"))
    best_desc_file = None
    for df in desc_files:
        basename = os.path.basename(df).replace("_description.md", "")
        if task_name and (basename.lower() == task_name.lower() or 
                          basename.lower().replace("-", "_") == task_name.lower().replace("-", "_") or
                          task_name.lower().startswith(basename.lower())):
            best_desc_file = df
            break
    
    if best_desc_file:
        with open(best_desc_file, "r") as f:
            desc_content = f.read()
        # Extract overview section
        overview_match = re.search(r"## overview\s*\n(.*?)(?=\n## |\Z)", desc_content, re.DOTALL)
        if overview_match:
            description = overview_match.group(1).strip()
        
        # Extract objective function  
        obj_match = re.search(r"## objective_function\s*\n(.*?)(?=\n## |\Z)", desc_content, re.DOTALL)
        objective_text = ""
        if obj_match:
            obj_section = obj_match.group(1)
            # Extract full_expression
            expr_match = re.search(r"\*\*full_expression\*\*:\s*(.+)", obj_section)
            if expr_match:
                objective_text = expr_match.group(1).strip()
    else:
        description = f"{dm['title_en']} inverse problem."
    
    # Find asset folder
    asset_folders = glob.glob(os.path.join(assets_dir, f"Task_{task_id_padded}_*"))
    if not asset_folders:
        asset_folders = glob.glob(os.path.join(assets_dir, f"Task_{task_id}_*"))
    
    has_vis = False
    vis_filename = "vis_result.png"
    if asset_folders:
        folder = asset_folders[0]
        folder_name = os.path.basename(folder)
        # Check for visualization
        for vf in ["vis_result.png", "reconstruction_result.png"]:
            if os.path.exists(os.path.join(folder, vf)):
                has_vis = True
                vis_filename = vf
                break
    else:
        folder_name = f"Task_{task_id_padded}_{task_name}"
    
    # Build task entry
    task_entry = {
        "id": task_id_padded,
        "id_num": task_id,
        "name": task_name,
        "domain": dm["domain"],
        "domain_name": dm["domain_name"],
        "title": dm["title_en"],
        "description": description[:500] if description else f"{dm['title_en']} - computational imaging inverse problem.",
        "metrics": {},
        "images": {
            "folder": folder_name,
            "vis_result": vis_filename if has_vis else None,
        }
    }
    
    if psnr is not None:
        task_entry["metrics"]["psnr"] = psnr
    if ssim is not None:
        task_entry["metrics"]["ssim"] = ssim
    if eval_type:
        task_entry["metrics"]["eval_type"] = eval_type
    
    tasks_db[f"task_{task_id_padded}"] = task_entry

# Build domain summary
domain_summary = {}
for key, label in domain_labels.items():
    tasks_in_domain = [t for t in tasks_db.values() if t["domain"] == key]
    domain_summary[key] = {
        **label,
        "task_count": len(tasks_in_domain),
        "task_ids": [t["id_num"] for t in sorted(tasks_in_domain, key=lambda x: x["id_num"])]
    }

output = {
    "meta": {
        "title": "Agent-Imaging: A Universal Agent for Computational Imaging Inverse Problems",
        "total_tasks": len(tasks_db),
        "total_domains": 8,
    },
    "domains": domain_summary,
    "tasks": tasks_db
}

# Write output
out_path = "/data/yjh/agent-imaging-website/public/data/tasks_db.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Generated tasks_db.json with {len(tasks_db)} tasks across 8 domains")
print(f"Domain counts: {json.dumps({k: v['task_count'] for k,v in domain_summary.items()})}")
