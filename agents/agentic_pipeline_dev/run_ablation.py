"""
Ablation Study Runner

Runs the same tasks under 4 different skill system configurations:
  Mode A: no_skill      — No persistent skill system (baseline)
  Mode B: instance_only — Only instance (few-shot) retrieval, frozen DB
  Mode C: experience_only — Only experience (pattern) retrieval, frozen DB
  Mode D: instance_exp  — Instance + Experience retrieval, frozen DB

Usage:
  # Run all 4 modes on all tasks
  python run_ablation.py

  # Run specific mode(s)
  python run_ablation.py --modes no_skill instance_exp

  # Run specific tasks (via env var)
  TASK_NAMES="deblur,denoise" python run_ablation.py

  # Run a specific mode on specific tasks
  python run_ablation.py --modes instance_only --tasks deblur denoise
"""

import os
import sys
import yaml
import time
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from openai import OpenAI

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_flow_test_only import InverseProblemWorkflowTestOnly
from persistent_skill_system.manager import SkillManager
from ablation_skill_manager import (
    NoSkillManager,
    InstanceOnlyManager,
    ExperienceOnlyManager,
    InstanceExpManager,
)

# Try importing reporter (optional)
try:
    from reporting import ExecutionReporter
except ImportError:
    ExecutionReporter = None

TASK_DESCRIPTION_BASE_DIR = "/data/yjh/task_descriptions"

# =============================================================================
# Mode definitions
# =============================================================================
ABLATION_MODES = {
    "no_skill": {
        "label": "A: No Skill System (Baseline)",
        "description": "No persistent skill system. Pure LLM without any retrieved knowledge.",
        "factory": lambda real_mgr: NoSkillManager(),
    },
    "instance_only": {
        "label": "B: Instance Only (Few-Shot)",
        "description": "Only instance/few-shot examples are injected. Frozen DB.",
        "factory": lambda real_mgr: InstanceOnlyManager(real_mgr),
    },
    "experience_only": {
        "label": "C: Experience Only (Patterns)",
        "description": "Only experience patterns are injected. Frozen DB.",
        "factory": lambda real_mgr: ExperienceOnlyManager(real_mgr),
    },
    "instance_exp": {
        "label": "D: Instance + Experience",
        "description": "Both instances and experiences are injected. Frozen DB.",
        "factory": lambda real_mgr: InstanceExpManager(real_mgr),
    },
}


# =============================================================================
# Helpers
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_task_description(task_name: str) -> str:
    desc_path = Path(TASK_DESCRIPTION_BASE_DIR) / f"{task_name}_description.md"
    if desc_path.exists():
        with open(desc_path, "r", encoding='utf-8') as f:
            return f.read()
    else:
        return f"Recover the signal from noisy measurements using a physics-based inverse solver. Task: {task_name}"


def run_single_task(
    task_info: dict,
    client: OpenAI,
    model_name: str,
    skill_manager,
    mode_key: str,
) -> Dict:
    """Execute a single task with the given skill manager variant."""
    task_name = task_info['name']
    start_time = time.time()

    print(f"\n  ▶ Task: {task_name} | Mode: {mode_key}")

    try:
        os.makedirs(task_info['working_folder'], exist_ok=True)
        task_description = load_task_description(task_name)

        # Use a mode-specific output directory to avoid collisions
        root_output = f"/data/yjh/end_sandbox/ablation_{mode_key}"

        workflow = InverseProblemWorkflowTestOnly(
            task_name=task_name,
            task_desc=task_description,
            gt_code_path=task_info['gt_code_path'],
            python_path=task_info.get('python_path', sys.executable),
            working_dir=task_info['working_folder'],
            client=client,
            model_name=model_name,
            root_output_dir=root_output,
            skill_manager=skill_manager,
        )

        success = workflow.run()
        elapsed = time.time() - start_time

        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  [{status}] {task_name} ({mode_key}) — {elapsed:.1f}s")

        return {
            "task_name": task_name,
            "mode": mode_key,
            "success": success,
            "elapsed_sec": round(elapsed, 2),
            "iterations": workflow.retry_count,
            "knowledge_used": len(workflow.used_knowledge_ids),
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ EXCEPTION {task_name} ({mode_key}): {error_msg}")
        traceback.print_exc()
        return {
            "task_name": task_name,
            "mode": mode_key,
            "success": False,
            "elapsed_sec": round(elapsed, 2),
            "iterations": -1,
            "knowledge_used": 0,
            "error": error_msg,
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for skill system")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(ABLATION_MODES.keys()),
        default=list(ABLATION_MODES.keys()),
        help="Which ablation modes to run (default: all 4)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names to run (default: all from config)",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Path to task config YAML (default: config/config_task_2.yaml)",
    )
    parser.add_argument(
        "--model",
        default="cds/Claude-4.6-opus",
        help="Model key from config_llm.yaml",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configs
    task_config_path = args.task_config or os.path.join(base_dir, "config", "config_task_2.yaml")
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")

    print("=" * 70)
    print("🔬 ABLATION STUDY RUNNER")
    print("=" * 70)
    print(f"Task Config : {task_config_path}")
    print(f"Model       : {args.model}")
    print(f"Modes       : {args.modes}")

    task_config = load_config(task_config_path)
    llm_config = load_config(llm_config_path)

    model_key = args.model
    if model_key not in llm_config['models']:
        raise ValueError(f"Model '{model_key}' not found in LLM config")

    model_conf = llm_config['models'][model_key]

    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url'],
    )

    # Initialize real SkillManager (will be wrapped by ablation variants)
    skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    real_skill_manager = None
    try:
        real_skill_manager = SkillManager(
            db_path=skill_db_path,
            client=client,
            model_name=model_key,
        )
        print(f"✓ Real SkillManager loaded from: {skill_db_path}")
    except Exception as e:
        print(f"⚠ Failed to load SkillManager: {e}")
        print("  Modes requiring DB (instance_only, experience_only, instance_exp) will fail.")

    # Determine tasks to run
    all_tasks = task_config.get('tasks', [])
    if args.tasks:
        tasks_to_run = [t for t in all_tasks if t['name'] in args.tasks]
        if not tasks_to_run:
            raise ValueError(f"No matching tasks for: {args.tasks}")
    else:
        # Also support TASK_NAMES env var
        env_filter = os.environ.get("TASK_NAMES", "").strip()
        if env_filter:
            selected = [n.strip() for n in env_filter.split(",") if n.strip()]
            tasks_to_run = [t for t in all_tasks if t['name'] in selected]
        else:
            tasks_to_run = all_tasks

    print(f"Tasks       : {[t['name'] for t in tasks_to_run]}")
    print("=" * 70)

    # ---- Run all mode×task combinations ----
    all_results: List[Dict] = []
    total_start = time.time()

    for mode_key in args.modes:
        mode_info = ABLATION_MODES[mode_key]
        print(f"\n{'─'*60}")
        print(f"🧪 Mode: {mode_info['label']}")
        print(f"   {mode_info['description']}")
        print(f"{'─'*60}")

        # Build the ablation skill manager
        if mode_key == "no_skill":
            ablation_mgr = mode_info["factory"](None)
        else:
            if real_skill_manager is None:
                print(f"  ⚠ Skipping mode '{mode_key}': SkillManager not available")
                continue
            ablation_mgr = mode_info["factory"](real_skill_manager)

        for idx, task_info in enumerate(tasks_to_run, 1):
            print(f"\n  [{idx}/{len(tasks_to_run)}]", end="")
            result = run_single_task(
                task_info=task_info,
                client=client,
                model_name=model_key,
                skill_manager=ablation_mgr,
                mode_key=mode_key,
            )
            all_results.append(result)

            # Brief pause between tasks
            if idx < len(tasks_to_run):
                time.sleep(1.0)

    total_elapsed = time.time() - total_start

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("📊 ABLATION STUDY RESULTS")
    print("=" * 70)

    # Group by mode
    for mode_key in args.modes:
        mode_results = [r for r in all_results if r['mode'] == mode_key]
        if not mode_results:
            continue

        successes = sum(1 for r in mode_results if r['success'])
        total = len(mode_results)
        avg_time = sum(r['elapsed_sec'] for r in mode_results) / total if total else 0
        avg_iters = sum(r['iterations'] for r in mode_results if r['iterations'] >= 0) / max(1, sum(1 for r in mode_results if r['iterations'] >= 0))

        label = ABLATION_MODES[mode_key]['label']
        print(f"\n  {label}")
        print(f"    Success Rate : {successes}/{total} ({100*successes/total:.1f}%)")
        print(f"    Avg Time     : {avg_time:.1f}s")
        print(f"    Avg Iters    : {avg_iters:.1f}")

        for r in mode_results:
            icon = "✓" if r['success'] else "✗"
            err = f" — {r['error'][:60]}" if r['error'] else ""
            print(f"      {icon} {r['task_name']}: {r['elapsed_sec']}s, iters={r['iterations']}, knowledge={r['knowledge_used']}{err}")

    print(f"\n  Total Time: {total_elapsed:.1f}s")

    # Save JSON report
    report_dir = os.path.join(base_dir, "ablation_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_key,
        "modes_tested": args.modes,
        "tasks_tested": [t['name'] for t in tasks_to_run],
        "total_elapsed_sec": round(total_elapsed, 2),
        "results": all_results,
        "summary": {},
    }

    for mode_key in args.modes:
        mode_results = [r for r in all_results if r['mode'] == mode_key]
        if not mode_results:
            continue
        successes = sum(1 for r in mode_results if r['success'])
        total = len(mode_results)
        report["summary"][mode_key] = {
            "success_count": successes,
            "total_count": total,
            "success_rate": round(successes / total, 4) if total else 0,
            "avg_elapsed_sec": round(sum(r['elapsed_sec'] for r in mode_results) / total, 2) if total else 0,
        }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  📄 Report saved to: {report_path}")
    print("=" * 70)

    # Exit code: 0 if all passed, 1 otherwise
    all_ok = all(r['success'] for r in all_results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
