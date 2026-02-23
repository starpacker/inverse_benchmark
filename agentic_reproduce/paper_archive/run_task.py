#!/usr/bin/env python3
"""
纯 Paper-Driven 模式：PDF/Markdown → Auto-Reproduce
完全移除 gt_code_path 和外部 task_description 依赖
"""
import os
import sys
import yaml
import time
import traceback
from pathlib import Path
from openai import OpenAI
from typing import List, Dict

# Add current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_flow import InverseProblemWorkflow  # 已重构为纯 paper-driven 模式

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_single_task(
    task_info: dict,
    model_conf: dict,
    client: OpenAI,
    model_name: str,
    base_dir: str
) -> Dict[str, any]:
    """执行单个任务：仅依赖 paper_md_path"""
    task_name = task_info['name']
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"▶ Starting Task: {task_name}")
    print(f"{'='*60}")
    print(f"  Paper MD Path: {task_info['paper_md_path']}")
    print(f"  Working Dir  : {task_info['working_folder']}")
    print(f"  Python Path  : {task_info.get('python_path', 'default')}")

    try:
        # 确保工作目录存在
        os.makedirs(task_info['working_folder'], exist_ok=True)
        
        # ✅ 关键变更：不再加载外部 task_description，完全由 workflow 自动生成
        # 初始化工作流（仅需 paper_md_path）
        workflow = InverseProblemWorkflow(
            task_name=task_name,
            paper_md_path=task_info['paper_md_path'],  # ✅ 唯一输入源
            python_path=task_info.get('python_path', sys.executable),
            working_dir=task_info['working_folder'],
            client=client,
            model_name=model_name,
            root_output_dir="/data/yjh/paper_sandbox"  # 专用目录避免污染
        )
        
        # 执行完整 pipeline: Paper → TaskDesc → Data → Solver → Eval
        success = workflow.run()
        elapsed = time.time() - start_time
        
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
        
        return {
            "task_name": task_name,
            "success": False,
            "elapsed_sec": round(elapsed, 2),
            "error": error_msg
        }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_config_path = os.path.join(base_dir, "config", "config_task_paper.yaml")  # ✅ 新配置文件
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")

    # 加载配置
    print("Loading configurations...")
    task_config = load_config(task_config_path)
    llm_config = load_config(llm_config_path)

    # 选择模型
    model_key = "cds/Claude-4.6-opus"  # 可根据需要调整
    
    if model_key not in llm_config['models']:
        raise ValueError(f"Model '{model_key}' not found in {llm_config_path}")
    
    model_conf = llm_config['models'][model_key]
    print(f"Using LLM Model: {model_key}")

    # 初始化客户端
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )

    # 确定待执行任务列表
    all_tasks = task_config.get('tasks', [])
    if not all_tasks:
        raise ValueError("No tasks found in config_task_paper.yaml. Expected format:\n"
                         "tasks:\n"
                         "  - name: 'admm_inpainting'\n"
                         "    paper_md_path: '/path/to/paper.md'\n"
                         "    working_folder: '/path/to/workspace'")

    # 支持任务过滤
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
        # ✅ 验证配置字段（关键：必须包含 paper_md_path）
        if 'paper_md_path' not in task_info:
            raise ValueError(
                f"Task '{task_info.get('name', 'unknown')}' missing required field 'paper_md_path'.\n"
                "Config format must be:\n"
                "tasks:\n"
                "  - name: 'task_name'\n"
                "    paper_md_path: '/path/to/paper.md'\n"
                "    working_folder: '/path/to/workspace'"
            )
        
        if not os.path.exists(task_info['paper_md_path']):
            raise FileNotFoundError(
                f"Paper Markdown not found for task '{task_info.get('name', 'unknown')}': "
                f"{task_info['paper_md_path']}"
            )
        
        print(f"\n[Task {idx}/{len(tasks_to_run)}] {task_info['name']}")
        result = run_single_task(task_info, model_conf, client, model_key, base_dir)
        results.append(result)
        
        # 任务间间隔避免 API 限流
        if idx < len(tasks_to_run):
            time.sleep(2.0)  # 增加到 2 秒更安全

    total_elapsed = time.time() - total_start

    # 生成汇总报告
    print("\n" + "="*60)
    print("EXECUTION SUMMARY (Paper-Driven Mode)")
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
    
    print("="*60)
    
    # 退出码
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == "__main__":
    main()