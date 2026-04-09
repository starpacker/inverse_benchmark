import argparse
import os
import json
import sqlite3
import pandas as pd
from tabulate import tabulate
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistent_skill_system.storage import SkillStorage

def get_db_path():
    # Default location relative to this script
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       "persistent_skill_system", "skills_new.db")

def list_skills(args):
    storage = SkillStorage(args.db_path)
    skills = storage.get_all_skills()
    
    if not skills:
        print("No skills found.")
        return

    data = []
    for s in skills:
        data.append({
            "Name": s['name'],
            "Principle (Short)": s['principle'][:50] + "...",
            "Sources": len(s['source']),
            "Version": s['version'],
            "Updated": pd.to_datetime(s['update_time'], unit='s').strftime('%Y-%m-%d %H:%M')
        })
    
    print(tabulate(data, headers="keys", tablefmt="grid"))

def list_trajectories(args):
    storage = SkillStorage(args.db_path)
    trajs = storage.get_trajectories(limit=args.limit)
    
    if not trajs:
        print("No trajectories found.")
        return

    data = []
    for t in trajs:
        data.append({
            "ID": t['id'],
            "Task": t['task_name'],
            "Outcome": t['outcome'],
            "Created": pd.to_datetime(t['create_time'], unit='s').strftime('%Y-%m-%d %H:%M')
        })
    
    print(tabulate(data, headers="keys", tablefmt="grid"))

def export_skills(args):
    storage = SkillStorage(args.db_path)
    skills = storage.get_all_skills()
    
    with open(args.output, 'w') as f:
        f.write("# Exported Skills Library\n\n")
        for s in skills:
            f.write(f"## {s['name']}\n")
            f.write(f"- **Principle**: {s['principle']}\n")
            f.write(f"- **Apply When**: {s['when_to_apply']}\n")
            f.write(f"- **Sources**: {len(s['source'])} trajectories\n")
            f.write(f"- **Version**: {s['version']}\n\n")
    
    print(f"Exported {len(skills)} skills to {args.output}")

def export_trajectory(args):
    storage = SkillStorage(args.db_path)
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM trajectories WHERE id = ?", (args.id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print(f"Trajectory {args.id} not found.")
        return
        
    data = dict(row)
    traj_json = json.loads(data['trajectory_json'])
    
    with open(args.output, 'w') as f:
        json.dump(traj_json, f, indent=2)
    
    print(f"Exported trajectory {args.id} to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Manage Persistent Skills and Trajectories")
    parser.add_argument("--db-path", default=get_db_path(), help="Path to skills.db")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List Skills
    list_parser = subparsers.add_parser("list-skills", help="List all skills")
    list_parser.set_defaults(func=list_skills)
    
    # List Trajectories
    traj_parser = subparsers.add_parser("list-trajs", help="List recent trajectories")
    traj_parser.add_argument("--limit", type=int, default=20, help="Limit number of rows")
    traj_parser.set_defaults(func=list_trajectories)
    
    # Export Skills
    exp_parser = subparsers.add_parser("export-skills", help="Export skills to Markdown")
    exp_parser.add_argument("--output", default="skills_export.md", help="Output file path")
    exp_parser.set_defaults(func=export_skills)
    
    # Export Trajectory
    exp_traj_parser = subparsers.add_parser("export-traj", help="Export a specific trajectory to JSON")
    exp_traj_parser.add_argument("--id", required=True, help="Trajectory ID")
    exp_traj_parser.add_argument("--output", default="trajectory_dump.json", help="Output file path")
    exp_traj_parser.set_defaults(func=export_trajectory)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
