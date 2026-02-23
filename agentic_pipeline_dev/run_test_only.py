import os
import sys
import yaml
import time
import traceback
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Optional

# Add current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_flow_test_only import InverseProblemWorkflowTestOnly
from persistent_skill_system.manager import SkillManager
from reporting import ExecutionReporter

TASK_DESCRIPTION_BASE_DIR = "/data/yjh/task_descriptions"  # 可后续提取为配置项

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_task_description(task_name: str) -> str:
    """加载任务描述，支持降级策略"""
    desc_path = Path(TASK_DESCRIPTION_BASE_DIR) / f"{task_name}_description.md"
    if desc_path.exists():
        print(f"  ✓ Loading task description from: {desc_path}")
        with open(desc_path, "r", encoding='utf-8') as f:
            return f.read()
    else:
        print(f"  ⚠ Warning: Task description not found at {desc_path}. Using default.")
        return f"Recover the signal from noisy measurements using a physics-based inverse solver. Task: {task_name}"

def run_single_task(
    task_info: dict,
    model_conf: dict,
    client: OpenAI,
    model_name: str,
    base_dir: str,
    skill_manager: Optional[SkillManager] = None,
    reporter: Optional[ExecutionReporter] = None
) -> Dict[str, any]:
    """执行单个任务并返回结构化结果"""
    task_name = task_info['name']
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"▶ Starting Task (TEST MODE - READ ONLY SKILLS): {task_name}")
    print(f"{'='*60}")
    print(f"  GT Code Path : {task_info['gt_code_path']}")
    print(f"  Working Dir  : {task_info['working_folder']}")
    print(f"  Python Path  : {task_info.get('python_path', 'default')}")

    try:
        # 确保工作目录存在
        os.makedirs(task_info['working_folder'], exist_ok=True)
        
        # 加载任务描述
        task_description = load_task_description(task_name)
    
        # 初始化工作流
        workflow = InverseProblemWorkflowTestOnly(
            task_name=task_name,
            task_desc=task_description,
            gt_code_path=task_info['gt_code_path'],
            python_path=task_info.get('python_path', sys.executable),
            working_dir=task_info['working_folder'],
            client=client,
            model_name=model_name,
            root_output_dir="/data/yjh/end_sandbox",
            skill_manager=skill_manager
        )
        
        # --- TEST MODE ---
        # No need for monkey patching anymore as we use InverseProblemWorkflowTestOnly
        
        # 执行
        success = workflow.run()
        elapsed = time.time() - start_time
        
        # Add to reporter
        if reporter:
            reporter.add_result(task_name, workflow, success, elapsed)
        
        result = {
            "task_name": task_name,
            "success": success,
            "elapsed_sec": round(elapsed, 2),
            "error": None
        }
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"\n  [{status}] Task '{task_name}' completed in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n  ✗ EXCEPTION in task '{task_name}': {error_msg}")
        print(f"  Traceback:\n{traceback.format_exc()}")
        
        # Add failed entry to reporter
        if reporter:
            class DummyWorkflow:
                retry_count = 0
                used_knowledge_ids = []
                failure_history = [{'ticket_assigned_to': 'System', 'analysis': str(e)}]
                distillation_stats = {}
            reporter.add_result(task_name, DummyWorkflow(), False, elapsed)
        
        return {
            "task_name": task_name,
            "success": False,
            "elapsed_sec": round(elapsed, 2),
            "error": error_msg
        }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_config_path = os.path.join(base_dir, "config", "config_task_2.yaml")
    # task_config_path = os.path.join(base_dir, "config", "config_task.yaml")
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")

    # 加载配置
    print("Loading configurations...")
    task_config = load_config(task_config_path)
    llm_config = load_config(llm_config_path)

    # 选择模型（固定使用指定模型，也可扩展为参数化）
    # model_key = "claude-opus-4-5-20251101-thinking"
    # model_key = "gpt-5.2-thinking"
    # model_key = "gemini-3-pro-preview"
    model_key = "cds/Claude-4.6-opus"
    
    if model_key not in llm_config['models']:
        raise ValueError(f"Model '{model_key}' not found in {llm_config_path}")
    
    model_conf = llm_config['models'][model_key]
    print(f"Using LLM Model: {model_key}")

    # 初始化客户端（所有任务复用同一客户端）
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )

    # Initialize Skill Manager (Persistent)
    # skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills.db")
    skill_db_path = os.path.join(base_dir, "persistent_skill_system", "skills_new.db")
    skill_manager = None
    try:
        skill_manager = SkillManager(
            db_path=skill_db_path,
            client=client,
            model_name=model_key
        )
        print(f"✓ Skill Manager initialized. DB Path: {skill_db_path}")
    except Exception as e:
        print(f"⚠ Failed to initialize Skill Manager: {e}")

    # Initialize Reporter
    reporter = ExecutionReporter(root_output_dir=base_dir)

    # 确定待执行任务列表
    all_tasks = task_config.get('tasks', [])
    if not all_tasks:
        raise ValueError("No tasks found in config_task.yaml")

    # 支持通过环境变量筛选任务子集：export TASK_NAMES="sim,deblur"
    task_filter = os.environ.get("TASK_NAMES", "").strip()
    if task_filter:
        selected_names = [name.strip() for name in task_filter.split(",") if name.strip()]
        tasks_to_run = [t for t in all_tasks if t['name'] in selected_names]
        if not tasks_to_run:
            raise ValueError(f"No matching tasks found for filter: {task_filter}")
        print(f"Filtered tasks to run ({len(tasks_to_run)}/{len(all_tasks)}): {selected_names}")
    else:
        tasks_to_run = all_tasks
        print(f"Running all tasks ({len(tasks_to_run)} total)")

    # 执行所有任务
    results: List[Dict] = []
    total_start = time.time()
    
    for idx, task_info in enumerate(tasks_to_run, 1):
        print(f"\n[Task {idx}/{len(tasks_to_run)}]")
        result = run_single_task(task_info, model_conf, client, model_key, base_dir, skill_manager=skill_manager, reporter=reporter)
        results.append(result)
        
        # 可选：任务间增加短暂间隔避免API限流
        if idx < len(tasks_to_run):
            time.sleep(1.0)

    total_elapsed = time.time() - total_start

    # 生成汇总报告
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY (READ-ONLY SKILLS)")
    print("="*60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Total Tasks   : {len(results)}")
    print(f"Successful    : {len(successful)} ✓")
    print(f"Failed        : {len(failed)} ✗")
    print(f"Total Time    : {total_elapsed:.2f}s")
    
    if failed:
        print("\nFailed Tasks Details:")
        for r in failed:
            print(f"  - {r['task_name']}: {r['error'] or 'Workflow returned False'}")
    
    # Generate JSON Report
    reporter.generate_report()
    
    print("="*60)
    
    # 整体退出码：存在失败任务则返回非零
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == "__main__":
    main()
