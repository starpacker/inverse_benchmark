import os
import yaml
import multiprocessing
import time
import sys
import traceback
from multiprocessing import Queue, Lock

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

def run_task_unit(model_name, gpu_id, split_index, output_root, base_dir):
    """执行单个任务单元：指定模型 + 指定GPU分片"""
    # 设置GPU可见性（关键：确保每个进程只看到分配的GPU）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 加载对应分片的任务配置
    config_path = os.path.join(base_dir, f"config/tasks_split_{split_index}.yaml")
    if not os.path.exists(config_path):
        safe_print(f"[GPU{gpu_id}] 配置文件不存在: {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tasks = config.get('tasks', [])
    
    if not tasks:
        safe_print(f"[{model_name} @ GPU{gpu_id}] 分片 {split_index} 无任务，跳过")
        return True

    safe_print(f"[{model_name} @ GPU{gpu_id}] 开始处理分片 {split_index} ({len(tasks)} 个任务)")
    
    # 创建模型专属输出目录
    model_output_dir = os.path.join(output_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    success_count = 0
    for task in tasks:
        task_name = task.get('name')
        gt_code = task.get('gt_code_path')
        working_folder = task.get('working_folder')
        
        if not all([task_name, gt_code, working_folder]):
            safe_print(f"[{model_name} @ GPU{gpu_id}] 无效任务配置: {task}")
            continue

        try:
            task_out_dir = os.path.join(model_output_dir, task_name)
            pipeline = InverseAgentPipeline(task_out_dir, model_name=model_name)
            pipeline.run(task_name, gt_code, working_folder)
            success_count += 1
        except Exception as e:
            safe_print(f"[{model_name} @ GPU{gpu_id}] 任务 {task_name} 失败: {str(e)}")
            traceback.print_exc()
    
    safe_print(f"[{model_name} @ GPU{gpu_id}] 分片 {split_index} 完成: {success_count}/{len(tasks)} 个任务成功")
    return success_count == len(tasks)

def worker_process(gpu_id, task_queue, output_root, base_dir):
    """GPU专属worker：持续消费任务队列"""
    safe_print(f"[GPU{gpu_id}] Worker 启动，绑定GPU {gpu_id}")
    
    while True:
        try:
            task_unit = task_queue.get(timeout=1.0)  # 1秒超时避免永久阻塞
            if task_unit is None:  # 收到终止信号
                safe_print(f"[GPU{gpu_id}] 收到终止信号，退出")
                break
                
            model_name, split_index = task_unit
            safe_print(f"[GPU{gpu_id}] 开始执行: {model_name} - split_{split_index}")
            
            success = run_task_unit(model_name, gpu_id, split_index, output_root, base_dir)
            
            status = "✓ 成功" if success else "✗ 部分失败"
            safe_print(f"[GPU{gpu_id}] 完成 {model_name} - split_{split_index} [{status}]")
            
        except multiprocessing.queues.Empty:
            # 队列空闲时检查是否所有任务已完成
            if task_queue.empty():
                safe_print(f"[GPU{gpu_id}] 任务队列已空，退出")
                break
        except Exception as e:
            safe_print(f"[GPU{gpu_id}] Worker 异常: {str(e)}")
            traceback.print_exc()
            break
    
    safe_print(f"[GPU{gpu_id}] Worker 退出")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = "/data/yjh/results_suite_final"
    num_gpus = 8  # 可根据实际GPU数量调整
    
    # 1. 创建任务队列：48个任务单元 (6模型 × 8分片)
    task_queue = Queue()
    for model in MODELS:
        for split_idx in range(num_gpus):
            task_queue.put((model, split_idx))
    
    # 2. 添加终止信号（每个worker一个None）
    for _ in range(num_gpus):
        task_queue.put(None)
    
    safe_print(f"✓ 初始化完成: {len(MODELS)} 个模型 × {num_gpus} 个GPU分片 = {task_queue.qsize() - num_gpus} 个任务单元")
    
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
        time.sleep(0.5)  # 避免日志交错
    
    # 4. 等待所有worker完成
    for p in processes:
        p.join()
    
    safe_print("\n" + "="*60)
    safe_print("✓ 所有模型的所有分片任务已完成")
    safe_print("="*60)

if __name__ == "__main__":
    # 设置多进程启动方法（Linux推荐spawn）
    if sys.platform.startswith('linux'):
        multiprocessing.set_start_method('spawn', force=True)
    main()
