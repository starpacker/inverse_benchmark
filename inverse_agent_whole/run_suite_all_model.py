import os
import yaml
import json
import multiprocessing
import time
import sys
import traceback
from multiprocessing import Queue, Lock
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import InverseAgentPipeline
from config import settings

# 所有要运行的模型列表
MODELS = [
    "gemini-3-pro-preview",
    "qwen3-max",
    "deepseek-v3.2",
    "gpt-5.2-thinking",
    "claude-opus-4-5-20251101-thinking",
    "kimi-k2-250905",
    "glm-4.7"
]

# 全局锁用于安全打印
print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()

def atomic_write_json(filepath, data):
    """原子写入JSON文件，避免多进程冲突"""
    tmp_path = f"{filepath}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, filepath)  # 原子替换
    except Exception as e:
        safe_print(f"[ERROR] 写入文件失败 {filepath}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_task_unit(model_name, gpu_id, split_index, output_root, base_dir):
    """执行单个任务单元：指定模型 + 指定GPU分片"""
    # 设置GPU可见性
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 创建临时结果目录
    temp_results_dir = os.path.join(output_root, "_temp_results", model_name)
    os.makedirs(temp_results_dir, exist_ok=True)
    
    # 加载对应分片的任务配置
    config_path = os.path.join(base_dir, f"config/tasks_split_{split_index}.yaml")
    if not os.path.exists(config_path):
        safe_print(f"[GPU{gpu_id}] 配置文件不存在: {config_path}")
        # 记录空结果
        atomic_write_json(
            os.path.join(temp_results_dir, f"split_{split_index}.json"),
            {"split_index": split_index, "tasks": {}, "error": "config_missing"}
        )
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        tasks = config.get('tasks', [])
    except Exception as e:
        safe_print(f"[GPU{gpu_id}] 无法加载配置 {config_path}: {e}")
        atomic_write_json(
            os.path.join(temp_results_dir, f"split_{split_index}.json"),
            {"split_index": split_index, "tasks": {}, "error": f"config_load_error: {str(e)}"}
        )
        return False

    if not tasks:
        safe_print(f"[{model_name} @ GPU{gpu_id}] 分片 {split_index} 无任务，跳过")
        atomic_write_json(
            os.path.join(temp_results_dir, f"split_{split_index}.json"),
            {"split_index": split_index, "tasks": {}, "note": "no_tasks"}
        )
        return True

    safe_print(f"[{model_name} @ GPU{gpu_id}] 开始处理分片 {split_index} ({len(tasks)} 个任务)")
    
    # 创建模型专属输出目录
    model_output_dir = os.path.join(output_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 记录任务结果
    task_results = {}
    start_time = time.time()
    
    for task_idx, task in enumerate(tasks, 1):
        task_name = task.get('name')
        gt_code = task.get('gt_code_path')
        working_folder = task.get('working_folder')
        
        if not all([task_name, gt_code, working_folder]):
            safe_print(f"[{model_name} @ GPU{gpu_id}] 无效任务配置: {task}")
            task_results[task_name] = {
                "success": False,
                "error": "invalid_config",
                "task_index": task_idx
            }
            continue

        task_start = time.time()
        try:
            task_out_dir = os.path.join(model_output_dir, task_name)
            pipeline = InverseAgentPipeline(task_out_dir, model_name=model_name)
            success = pipeline.run(task_name, gt_code, working_folder)
            
            task_results[task_name] = {
                "success": success,
                "duration_sec": round(time.time() - task_start, 2),
                "task_index": task_idx,
                "output_dir": task_out_dir
            }
            
            status = "✓" if success else "✗"
            safe_print(f"[{model_name} @ GPU{gpu_id}] [{task_idx}/{len(tasks)}] {status} {task_name} ({task_results[task_name]['duration_sec']}s)")
            
        except Exception as e:
            safe_print(f"[{model_name} @ GPU{gpu_id}] 任务 {task_name} 失败: {str(e)}")
            traceback.print_exc()
            task_results[task_name] = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration_sec": round(time.time() - task_start, 2),
                "task_index": task_idx
            }
    
    duration_total = round(time.time() - start_time, 2)
    success_count = sum(1 for r in task_results.values() if r.get("success", False))
    
    # 保存分片结果
    split_result = {
        "model": model_name,
        "gpu_id": gpu_id,
        "split_index": split_index,
        "total_tasks": len(tasks),
        "success_tasks": success_count,
        "duration_sec": duration_total,
        "timestamp": datetime.now().isoformat(),
        "tasks": task_results
    }
    
    atomic_write_json(
        os.path.join(temp_results_dir, f"split_{split_index}.json"),
        split_result
    )
    
    safe_print(f"[{model_name} @ GPU{gpu_id}] 分片 {split_index} 完成: {success_count}/{len(tasks)} 成功 ({duration_total}s)")
    return success_count == len(tasks)

def worker_process(gpu_id, task_queue, output_root, base_dir):
    """GPU专属worker：持续消费任务队列"""
    safe_print(f"[GPU{gpu_id}] Worker 启动，绑定GPU {gpu_id}")
    
    while True:
        try:
            task_unit = task_queue.get(timeout=1.0)
            if task_unit is None:  # 收到终止信号
                safe_print(f"[GPU{gpu_id}] 收到终止信号，退出")
                break
                
            model_name, split_index = task_unit
            safe_print(f"[GPU{gpu_id}] 开始执行: {model_name} - split_{split_index}")
            
            success = run_task_unit(model_name, gpu_id, split_index, output_root, base_dir)
            
            status = "✓ 全部成功" if success else "⚠ 部分失败"
            safe_print(f"[GPU{gpu_id}] 完成 {model_name} - split_{split_index} [{status}]")
            
        except multiprocessing.queues.Empty:
            if task_queue.empty():
                safe_print(f"[GPU{gpu_id}] 任务队列已空，退出")
                break
        except Exception as e:
            safe_print(f"[GPU{gpu_id}] Worker 异常: {str(e)}")
            traceback.print_exc()
            break
    
    safe_print(f"[GPU{gpu_id}] Worker 退出")

def generate_model_report(model_name, output_root, num_splits=8):
    """为单个模型生成汇总报告"""
    temp_dir = os.path.join(output_root, "_temp_results", model_name)
    if not os.path.exists(temp_dir):
        safe_print(f"[REPORT] 警告: 模型 {model_name} 无临时结果目录")
        return False
    
    # 收集所有分片结果
    all_tasks = {}
    split_results = []
    total_duration = 0
    
    for split_idx in range(num_splits):
        result_file = os.path.join(temp_dir, f"split_{split_idx}.json")
        if not os.path.exists(result_file):
            safe_print(f"[REPORT] 警告: 模型 {model_name} 分片 {split_idx} 结果缺失")
            continue
            
        try:
            with open(result_file, 'r') as f:
                split_data = json.load(f)
                split_results.append(split_data)
                total_duration += split_data.get("duration_sec", 0)
                
                # 合并任务结果（任务名唯一）
                for task_name, task_result in split_data.get("tasks", {}).items():
                    if task_name in all_tasks:
                        safe_print(f"[REPORT] 警告: 任务名重复 {task_name} (模型: {model_name})")
                    all_tasks[task_name] = task_result
        except Exception as e:
            safe_print(f"[REPORT] 读取分片 {split_idx} 结果失败: {e}")
    
    if not all_tasks:
        safe_print(f"[REPORT] 模型 {model_name} 无有效任务结果")
        return False
    
    # 计算统计信息
    total_tasks = len(all_tasks)
    success_tasks = sum(1 for r in all_tasks.values() if r.get("success", False))
    accuracy = success_tasks / total_tasks if total_tasks > 0 else 0
    
    # 按任务索引排序
    sorted_tasks = dict(sorted(
        all_tasks.items(), 
        key=lambda x: x[1].get("task_index", 999)
    ))
    
    # 生成报告
    report = {
        "model": model_name,
        "report_generated_at": datetime.now().isoformat(),
        "total_splits_processed": len(split_results),
        "total_tasks": total_tasks,
        "success_tasks": success_tasks,
        "failed_tasks": total_tasks - success_tasks,
        "accuracy": round(accuracy, 4),
        "accuracy_percentage": f"{accuracy*100:.2f}%",
        "total_duration_sec": round(total_duration, 2),
        "average_task_duration_sec": round(total_duration / total_tasks, 2) if total_tasks > 0 else 0,
        "task_details": sorted_tasks,
        "split_summary": [
            {
                "split_index": s.get("split_index"),
                "gpu_id": s.get("gpu_id"),
                "tasks_in_split": s.get("total_tasks", 0),
                "success_in_split": s.get("success_tasks", 0),
                "duration_sec": s.get("duration_sec", 0)
            }
            for s in split_results
        ]
    }
    
    # 保存报告
    report_path = os.path.join(output_root, f"{model_name}_report.json")
    atomic_write_json(report_path, report)
    
    # 生成简洁文本摘要
    summary_path = os.path.join(output_root, f"{model_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"生成时间: {report['report_generated_at']}\n")
        f.write(f"总任务数: {total_tasks}\n")
        f.write(f"成功任务: {success_tasks}\n")
        f.write(f"失败任务: {total_tasks - success_tasks}\n")
        f.write(f"准确率: {report['accuracy_percentage']}\n")
        f.write(f"总耗时: {total_duration:.2f} 秒\n")
        f.write(f"平均任务耗时: {report['average_task_duration_sec']:.2f} 秒\n\n")
        f.write("失败任务详情:\n")
        for name, res in sorted_tasks.items():
            if not res.get("success", False):
                err = res.get("error", "unknown_error")
                f.write(f"  ✗ {name}: {err[:100]}\n")
    
    safe_print(f"[REPORT] ✓ 模型 {model_name} 报告生成: {success_tasks}/{total_tasks} ({report['accuracy_percentage']})")
    safe_print(f"         详细报告: {report_path}")
    safe_print(f"         摘要: {summary_path}")
    
    return True

def generate_overall_summary(output_root, models):
    """生成所有模型的对比摘要"""
    summary_data = []
    
    for model in models:
        report_path = os.path.join(output_root, f"{model}_report.json")
        if not os.path.exists(report_path):
            summary_data.append({
                "model": model,
                "status": "no_report",
                "accuracy": 0,
                "total_tasks": 0,
                "success_tasks": 0
            })
            continue
            
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
                summary_data.append({
                    "model": model,
                    "status": "completed",
                    "accuracy": report["accuracy"],
                    "accuracy_percentage": report["accuracy_percentage"],
                    "total_tasks": report["total_tasks"],
                    "success_tasks": report["success_tasks"],
                    "duration_sec": report["total_duration_sec"]
                })
        except Exception as e:
            safe_print(f"[SUMMARY] 读取模型 {model} 报告失败: {e}")
            summary_data.append({
                "model": model,
                "status": "report_error",
                "accuracy": 0,
                "total_tasks": 0,
                "success_tasks": 0
            })
    
    # 按准确率排序
    summary_data.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
    
    # 生成CSV格式摘要
    csv_path = os.path.join(output_root, "models_comparison.csv")
    with open(csv_path, 'w') as f:
        f.write("Model,Status,Accuracy,Success/Total,Duration(sec)\n")
        for item in summary_data:
            if item["status"] == "completed":
                f.write(f"{item['model']},{item['status']},{item['accuracy_percentage']},{item['success_tasks']}/{item['total_tasks']},{item['duration_sec']:.2f}\n")
            else:
                f.write(f"{item['model']},{item['status']},N/A,0/0,0.00\n")
    
    # 生成Markdown格式摘要
    md_path = os.path.join(output_root, "models_comparison.md")
    with open(md_path, 'w') as f:
        f.write("# 模型性能对比摘要\n\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
        f.write("| 模型 | 状态 | 准确率 | 成功/总数 | 总耗时(秒) |\n")
        f.write("|------|------|--------|-----------|------------|\n")
        for item in summary_data:
            if item["status"] == "completed":
                f.write(f"| `{item['model']}` | ✅ 完成 | **{item['accuracy_percentage']}** | {item['success_tasks']}/{item['total_tasks']} | {item['duration_sec']:.2f} |\n")
            else:
                f.write(f"| `{item['model']}` | ❌ {item['status']} | N/A | 0/0 | 0.00 |\n")
    
    safe_print("\n" + "="*70)
    safe_print("✓ 所有模型报告生成完成")
    safe_print("="*70)
    safe_print(f"📊 模型对比摘要: {csv_path}")
    safe_print(f"📝 Markdown报告: {md_path}")
    safe_print("="*70)
    
    # 控制台输出简洁排名
    print("\n🏆 模型准确率排名:")
    for i, item in enumerate(summary_data[:5], 1):  # Top 5
        if item["status"] == "completed":
            print(f"  {i}. {item['model']:<30} {item['accuracy_percentage']}")
    print("="*70)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = "/data/yjh/results_suite_final"
    num_gpus = 8  # 可根据实际GPU数量调整
    
    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)
    
    # 1. 创建任务队列
    task_queue = Queue()
    total_units = 0
    for model in MODELS:
        for split_idx in range(num_gpus):
            task_queue.put((model, split_idx))
            total_units += 1
    
    # 2. 添加终止信号
    for _ in range(num_gpus):
        task_queue.put(None)
    
    safe_print(f"✓ 初始化完成: {len(MODELS)} 个模型 × {num_gpus} 个GPU分片 = {total_units} 个任务单元")
    safe_print(f"  输出目录: {output_root}")
    safe_print(f"  临时结果: {os.path.join(output_root, '_temp_results')}")
    
    # 3. 启动GPU专属worker进程
    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(
            target=worker_process,
            args=(gpu_id, task_queue, output_root, base_dir),
            name=f"Worker-GPU{gpu_id}"
        )
        p.start()
        processes.append(p)
        safe_print(f"→ 启动 Worker-GPU{gpu_id} (PID: {p.pid})")
        time.sleep(0.3)  # 避免日志交错
    
    # 4. 等待所有worker完成
    for p in processes:
        p.join()
    
    safe_print("\n" + "="*70)
    safe_print("✓ 所有任务单元执行完成，开始生成模型报告...")
    safe_print("="*70)
    
    # 5. 为每个模型生成汇总报告
    for model in MODELS:
        generate_model_report(model, output_root, num_splits=num_gpus)
    
    # 6. 生成整体对比摘要
    generate_overall_summary(output_root, MODELS)
    
    # 7. （可选）清理临时文件
    # temp_dir = os.path.join(output_root, "_temp_results")
    # if os.path.exists(temp_dir):
    #     import shutil
    #     shutil.rmtree(temp_dir)
    #     safe_print(f"✓ 已清理临时结果目录: {temp_dir}")

if __name__ == "__main__":
    # 设置多进程启动方法
    if sys.platform.startswith('linux'):
        multiprocessing.set_start_method('spawn', force=True)
    main()