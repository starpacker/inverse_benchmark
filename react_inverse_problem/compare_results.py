import os
import json
import yaml
import sys

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_report(report_path):
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def main():
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    config_path = "config/config.yaml"
    # Assuming the script is run from project root, or we can use relative path
    sandbox_root = "/data/yjh/react_inverse_problem_sandboxes"
    
    # Defaults
    model_base = "cds/Claude-4.6-opus_old"
    model_new = "cds/Claude-4.6-opus"
    
    if len(sys.argv) > 1:
        model_base = sys.argv[1]
    if len(sys.argv) > 2:
        model_new = sys.argv[2]

    # Load config to get task names
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    tasks = config.get('tasks', [])
    
    table_data = []
    headers = ["Task", "Base Success", "New Success", "Diff", "Rounds (Base/New)"]
    
    total_base_success = 0
    total_new_success = 0
    total_questions = 0
    
    for task in tasks:
        task_name = task['name']
        
        # Paths
        path_base = os.path.join(sandbox_root, model_base, task_name, "task_report.json")
        path_new = os.path.join(sandbox_root, model_new, task_name, "task_report.json")
        
        report_base = load_report(path_base)
        report_new = load_report(path_new)
        
        # Stats
        base_s = 0
        new_s = 0
        base_q = 0
        new_q = 0
        base_rounds = "N/A"
        new_rounds = "N/A"
        
        if report_base:
            base_s = report_base["results"].get("SUCCESS", 0)
            base_q = report_base.get("processed_questions", 0)
            # average rounds
            details = report_base.get("details", [])
            if details:
                base_rounds = f"{sum(d['rounds_used'] for d in details) / len(details):.1f}"
        
        if report_new:
            new_s = report_new["results"].get("SUCCESS", 0)
            new_q = report_new.get("processed_questions", 0)
            details = report_new.get("details", [])
            if details:
                new_rounds = f"{sum(d['rounds_used'] for d in details) / len(details):.1f}"
        
        # Use whichever is non-zero, or just max
        q_count = max(base_q, new_q)
        
        if q_count > 0:
            total_questions += q_count
            total_base_success += base_s
            total_new_success += new_s
            
            diff = new_s - base_s
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            
            base_disp = f"{base_s}/{q_count}"
            new_disp = f"{new_s}/{q_count}"
            
            table_data.append([task_name, base_disp, new_disp, diff_str, f"{base_rounds} / {new_rounds}"])
        else:
            table_data.append([task_name, "N/A", "N/A", "-", "-"])

    print("\n" + "="*60)
    print(f"COMPARISON REPORT")
    print(f"Base Model: {model_base}")
    print(f"New Model:  {model_new}")
    print("="*60 + "\n")
    
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        # Simple fallback
        print(f"{'Task':<20} | {'Base':<10} | {'New':<10} | {'Diff':<5}")
        print("-" * 55)
        for row in table_data:
            print(f"{row[0]:<20} | {row[1]:<10} | {row[2]:<10} | {row[3]:<5}")
            
    if total_questions > 0:
        base_rate = (total_base_success / total_questions) * 100
        new_rate = (total_new_success / total_questions) * 100
        imp = new_rate - base_rate
        
        print("\nSUMMARY:")
        print(f"Total Questions: {total_questions}")
        print(f"Base Success:    {total_base_success} ({base_rate:.2f}%)")
        print(f"New Success:     {total_new_success}  ({new_rate:.2f}%)")
        print(f"Improvement:     {imp:+.2f}%")
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    main()
