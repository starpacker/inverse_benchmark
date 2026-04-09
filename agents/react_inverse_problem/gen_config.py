import os
import glob
import yaml
from pathlib import Path

def main():
    input_yaml = "config/config_selected.yaml"
    output_yaml = "config/gt_code_index.yaml"

    with open(input_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    gt_index = {
        "global": config.get("global", {}),
        "tasks": []
    }

    for task in config["tasks"]:
        name = task["name"]
        working_folder = task["working_folder"]

        # 标准化为绝对路径（可选但推荐）
        working_folder_abs = os.path.abspath(working_folder)

        if not os.path.isdir(working_folder_abs):
            print(f"[WARN] Working folder not found: {name} -> {working_folder_abs}")
            gt_index["tasks"].append({
                "name": name,
                "working_folder": working_folder,  # 保留原始值
                "gt_code_path": None,
                "status": "folder_missing"
            })
            continue

        # 查找 *_code.py
        pattern = os.path.join(working_folder_abs, "*_code.py")
        matches = glob.glob(pattern)

        if len(matches) == 0:
            print(f"[MISS] No *_code.py found for task: {name} (dir: {working_folder_abs})")
            gt_index["tasks"].append({
                "name": name,
                "working_folder": working_folder,
                "gt_code_path": None,
                "status": "no_code_file"
            })
        elif len(matches) > 1:
            print(f"[CONFLICT] Multiple *_code.py found for task: {name}")
            for m in matches:
                print(f"    - {m}")
            gt_index["tasks"].append({
                "name": name,
                "working_folder": working_folder,
                "gt_code_path": [os.path.abspath(m) for m in matches],
                "status": "multiple_files"
            })
        else:
            code_path = os.path.abspath(matches[0])
            gt_index["tasks"].append({
                "name": name,
                "working_folder": working_folder,
                "gt_code_path": code_path,
                "status": "success"
            })

    # 写入输出文件
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(gt_index, f, allow_unicode=True, default_flow_style=False, indent=2)

    print(f"\n✅ Index saved to: {output_yaml}")
    print("⚠️  Tasks with status != 'success' need manual inspection.")

if __name__ == "__main__":
    main()