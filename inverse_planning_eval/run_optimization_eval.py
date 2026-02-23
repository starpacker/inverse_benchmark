#!/usr/bin/env python3
"""
Prompt-Centric Evaluation Pipeline
固定Claude模型作为执行器，对10轮优化prompt进行统一ELO tournament评估
"""
import os
import sys
import json
import yaml
import glob
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set HF Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Import project modules
from evaluation.similarity import SimilarityEvaluator
from evaluation.elo import EloEvaluator
from plan_generator import PlanGenerator

# Setup root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_dataset(task_config_path):
    """加载任务数据集"""
    config_data = load_yaml(task_config_path)
    tasks_list = config_data.get('tasks', [])
    dataset = {}
    golden_plan_base = "/data/yjh/golden_plans"
    # golden_plan_base = "/home/yjh/inverse_planning_eval/data/golden_plans"  # 根据实际路径调整
    
    for task_entry in tasks_list:
        task_name = task_entry.get('name')
        gt_code_path = task_entry.get('gt_code_path')
        
        if not gt_code_path or not os.path.exists(gt_code_path):
            logger.warning(f"GT Code not found for {task_name}: {gt_code_path}")
            continue
            
        with open(gt_code_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            
        plan_path = os.path.join(golden_plan_base, f"{task_name}_golden_plan.json")
        if not os.path.exists(plan_path):
            logger.warning(f"Golden Plan not found for {task_name}: {plan_path}")
            continue
            
        with open(plan_path, 'r', encoding='utf-8') as f:
            golden_plan = json.load(f)
            
        dataset[task_name] = {
            'code': code_content,
            'golden_plan': golden_plan,
            'golden_plan_str': json.dumps(golden_plan, indent=2, ensure_ascii=False)
        }
    return dataset

def load_all_prompts(prompt_dir):
    """按epoch顺序加载10个优化prompt"""
    prompt_files = sorted(
        glob.glob(os.path.join(prompt_dir, "prompt_epoch_??.txt")),
        key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0])
    )
    
    if len(prompt_files) != 6:
        logger.warning(f"Expected 6 prompt files, found {len(prompt_files)} in {prompt_dir}")
        # Not raising error, just warning
    
    prompts = {}
    for i, path in enumerate(prompt_files, 1):
        with open(path, 'r', encoding='utf-8') as f:
            prompts[f"epoch_{i:02d}"] = f.read().strip()
    
    logger.info(f"Loaded {len(prompts)} prompts in order: {list(prompts.keys())}")
    return prompts

def evaluate_single_task(task_name, task_data, prompts, config_path, judge_config, output_base_dir):
    """
    对单个任务评估所有10个prompt
    
    Returns:
        dict: {
            "epoch_ratings": { "epoch_01": 1520, ... },
            "golden_rating": 1500,
            "similarity_scores": { "epoch_01": { "cosine": 0.85, ... }, ... },
            "plans_archive": { "epoch_01": "plan text", ... }
        }
    """
    task_dir = os.path.join(output_base_dir, "task_logs", task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Setup task-specific logger
    task_log_file = os.path.join(task_dir, "evaluation.log")
    file_handler = logging.FileHandler(task_log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    try:
        logger.info(f"\n{'='*60}\nEvaluating Task: {task_name}\n{'='*60}")
        logger.info(f"Code snippet (first 200 chars): {task_data['code'][:100]}...")
        
        # Step 1: 用Claude生成所有10个prompt对应的计划
        logger.info("\n[Step 1] Generating plans with Claude Opus for all 10 prompts...")
        plans_by_epoch = {}
        similarity_evaluator = SimilarityEvaluator()
        similarity_scores = {}
        
        for epoch_id, prompt_text in prompts.items():
            logger.info(f"  Generating plan for {epoch_id}...")
            
            # 创建专用generator（每个prompt独立实例）
            generator = PlanGenerator(config_path, prompt_text)
            model_key = "claude-opus-4-5-20251101-thinking"
            
            # Phase 1: Description Generation (Code -> Description)
            description_text = generator.generate_description(
                model_key=model_key,
                code_content=task_data['code'],
                task_name=task_name,
                use_cache=False # Force regenerate with new prompt
            )
            
            if not description_text:
                logger.error(f"    ✗ Failed to generate description for {epoch_id}")
                continue
                
            # Phase 2: Plan Generation (Description -> Plan)
            plan_text = generator.generate_plan(
                model_key=model_key,
                description=description_text
            )
            
            if not plan_text or len(plan_text.strip()) < 50:
                logger.error(f"    ✗ Failed to generate valid plan for {epoch_id}")
                continue
            
            # 保存计划归档
            plan_path = os.path.join(task_dir, f"{epoch_id}_plan.txt")
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(plan_text)
            
            # Also save description for debugging
            desc_path = os.path.join(task_dir, f"{epoch_id}_desc.md")
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description_text)
                
            logger.info(f"    ✓ Plan saved to {os.path.basename(plan_path)}")
            
            plans_by_epoch[epoch_id] = plan_text
            
            # 计算相似度
            try:
                sim_metrics = similarity_evaluator.evaluate(plan_text, task_data['golden_plan'])
                similarity_scores[epoch_id] = sim_metrics
                logger.info(f"    Similarity: cosine={sim_metrics.get('cosine', 0):.4f}, jaccard={sim_metrics.get('jaccard', 0):.4f}")
            except Exception as e:
                logger.warning(f"    Similarity evaluation failed: {e}")
                similarity_scores[epoch_id] = {"cosine": 0, "jaccard": 0, "error": str(e)}
        
        # Step 2: 构建ELO tournament参赛者（10个prompt + Golden）
        logger.info("\n[Step 2] Building ELO tournament participants...")
        tournament_participants = {
            "Golden": task_data['golden_plan_str']  # 作为锚点
        }
        tournament_participants.update(plans_by_epoch)
        
        logger.info(f"  Tournament participants: {list(tournament_participants.keys())}")
        
        # Step 3: 运行统一ELO tournament
        logger.info("\n[Step 3] Running unified ELO tournament (all prompts + Golden)...")
        elo_evaluator = EloEvaluator(judge_config=judge_config)
        
        # 任务上下文（用于judge参考）
        task_context = f"Task: {task_name}\nCode snippet:\n{task_data['code']}..."
        
        try:
            ratings, match_history = elo_evaluator.run_tournament(
                tournament_participants, 
                task_context
            )
            
            # 保存tournament详情
            with open(os.path.join(task_dir, "elo_ratings.json"), 'w', encoding='utf-8') as f:
                json.dump(ratings, f, indent=2, ensure_ascii=False)
            
            with open(os.path.join(task_dir, "match_history.json"), 'w', encoding='utf-8') as f:
                json.dump(match_history, f, indent=2, ensure_ascii=False)
            
            logger.info("  ELO Ratings:")
            for name, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
                marker = "🌟" if name == "Golden" else "  "
                logger.info(f"    {marker} {name:12s}: {rating:.1f}")
            
            # 提取epoch ratings（排除Golden）
            epoch_ratings = {
                k: v for k, v in ratings.items() 
                if k.startswith("epoch_")
            }
            
            # 确保10个epoch都有rating（缺失的设为最低值）
            for epoch_id in prompts.keys():
                if epoch_id not in epoch_ratings:
                    epoch_ratings[epoch_id] = min(ratings.values()) - 100
            
        except Exception as e:
            logger.error(f"  ✗ ELO tournament failed: {e}")
            # 回退到基于相似度的伪rating
            max_sim = max([s.get('cosine', 0) for s in similarity_scores.values()] + [0.1])
            epoch_ratings = {
                eid: 1500 + 200 * (sim.get('cosine', 0) / max_sim) 
                for eid, sim in similarity_scores.items()
            }
            ratings = {"Golden": 1500, **epoch_ratings}
        
        # Step 4: 保存完整结果
        task_result = {
            "task_name": task_name,
            "golden_rating": ratings.get("Golden", 1500),
            "epoch_ratings": epoch_ratings,
            "similarity_scores": similarity_scores,
            "plans_summary": {eid: p[:100] + "..." for eid, p in plans_by_epoch.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(task_dir, "task_result.json"), 'w', encoding='utf-8') as f:
            json.dump(task_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Task {task_name} evaluation completed. Results in {task_dir}")
        return task_result
        
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

def aggregate_results(all_task_results, output_dir):
    """聚合所有任务结果，生成跨epoch统计和可视化"""
    # 按epoch组织数据
    epoch_data = {f"epoch_{i:02d}": {"ratings": [], "cosine": [], "jaccard": []} for i in range(1, 11)}
    
    for task_res in all_task_results:
        for epoch_id in epoch_data.keys():
            if epoch_id in task_res["epoch_ratings"]:
                epoch_data[epoch_id]["ratings"].append(task_res["epoch_ratings"][epoch_id])
            if epoch_id in task_res["similarity_scores"]:
                sim = task_res["similarity_scores"][epoch_id]
                epoch_data[epoch_id]["cosine"].append(sim.get('cosine', 0))
                epoch_data[epoch_id]["jaccard"].append(sim.get('jaccard', 0))
    
    # 计算平均值和标准差
    epochs = sorted(epoch_data.keys())
    avg_ratings = []
    std_ratings = []
    avg_cosine = []
    std_cosine = []
    
    for eid in epochs:
        ratings = epoch_data[eid]["ratings"]
        cosine = epoch_data[eid]["cosine"]
        
        avg_ratings.append(np.mean(ratings) if ratings else 0)
        std_ratings.append(np.std(ratings) if len(ratings) > 1 else 0)
        avg_cosine.append(np.mean(cosine) if cosine else 0)
        std_cosine.append(np.std(cosine) if len(cosine) > 1 else 0)
    
    # 保存聚合结果
    agg_result = {
        "epochs": epochs,
        "avg_ratings": avg_ratings,
        "std_ratings": std_ratings,
        "avg_cosine": avg_cosine,
        "std_cosine": std_cosine,
        "task_count": len(all_task_results),
        "timestamp": datetime.now().isoformat()
    }
    
    agg_path = os.path.join(output_dir, "aggregated_results.json")
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(agg_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Aggregated results saved to {agg_path}")
    
    # 生成可视化
    plot_results(epochs, avg_ratings, std_ratings, avg_cosine, std_cosine, output_dir)
    
    return agg_result

def plot_results(epochs, avg_ratings, std_ratings, avg_cosine, std_cosine, output_dir):
    """生成双Y轴图表：ELO rating + Cosine similarity"""
    x = np.arange(len(epochs))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左Y轴：ELO Rating
    color1 = '#1f77b4'
    ax1.set_xlabel('Optimization Epoch', fontsize=12)
    ax1.set_ylabel('ELO Rating (vs Golden)', color=color1, fontsize=12)
    ax1.plot(x, avg_ratings, 'o-', color=color1, label='Avg ELO Rating', linewidth=2, markersize=8)
    ax1.fill_between(x, 
                     np.array(avg_ratings) - np.array(std_ratings),
                     np.array(avg_ratings) + np.array(std_ratings),
                     alpha=0.2, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.split('_')[1] for e in epochs])
    
    # 右Y轴：Cosine Similarity
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Cosine Similarity', color=color2, fontsize=12)
    ax2.plot(x, avg_cosine, 's--', color=color2, label='Avg Cosine Similarity', linewidth=2, markersize=8)
    ax2.fill_between(x,
                     np.array(avg_cosine) - np.array(std_cosine),
                     np.array(avg_cosine) + np.array(std_cosine),
                     alpha=0.2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 标题和图例
    plt.title('Prompt Optimization Progress: ELO Rating & Similarity vs Epoch', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'prompt_optimization_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {plot_path}")
    
    # 生成简明文本报告
    report = "\n" + "="*70 + "\n"
    report += "PROMPT OPTIMIZATION PROGRESS REPORT\n"
    report += "="*70 + "\n"
    report += f"{'Epoch':<10} {'Avg Rating':<15} {'Δ vs Prev':<15} {'Avg Cosine':<15}\n"
    report += "-"*70 + "\n"
    
    for i, eid in enumerate(epochs):
        delta = f"+{avg_ratings[i]-avg_ratings[i-1]:.1f}" if i > 0 else "—"
        report += f"{eid:<10} {avg_ratings[i]:<15.1f} {delta:<15} {avg_cosine[i]:<15.4f}\n"
    
    report += "="*70 + "\n"
    report += f"✓ Best epoch by ELO:   {epochs[np.argmax(avg_ratings)]} (rating={max(avg_ratings):.1f})\n"
    report += f"✓ Best epoch by Cosine: {epochs[np.argmax(avg_cosine)]} (similarity={max(avg_cosine):.4f})\n"
    report += "="*70 + "\n"
    
    logger.info(report)
    
    # 保存报告
    with open(os.path.join(output_dir, 'optimization_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description="Prompt-Centric Evaluation Pipeline")
    parser.add_argument("--prompt_dir", type=str, 
                       default="/home/yjh/inverse_planning_eval/optimized_prompts",
                       help="Directory containing prompt_epoch_??.txt files")
    parser.add_argument("--output_dir", type=str,
                       default="/home/yjh/inverse_planning_eval/evaluation_results",
                       help="Base directory for evaluation results")
    parser.add_argument("--task_config", type=str,
                       default="/home/yjh/inverse_planning_eval/config/gt_code_index.yaml",
                       help="Path to task configuration YAML")
    parser.add_argument("--config_path", type=str,
                       default="/home/yjh/inverse_planning_eval/config/config2.yaml",
                       help="Path to model configuration YAML")
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="Maximum number of tasks to evaluate (for debugging)")
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Setup main log file
    main_log_file = os.path.join(output_base_dir, "main_evaluation.log")
    file_handler = logging.FileHandler(main_log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*70)
    logger.info("PROMPT-CENTRIC EVALUATION PIPELINE STARTED")
    logger.info("="*70)
    logger.info(f"Prompt directory: {args.prompt_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Task config:      {args.task_config}")
    logger.info(f"Model config:     {args.config_path}")
    
    try:
        # Step 1: 加载10个优化prompt
        logger.info("\n[1/4] Loading optimized prompts...")
        prompts = load_all_prompts(args.prompt_dir)
        
        # Step 2: 加载任务数据集
        logger.info("\n[2/4] Loading task dataset...")
        dataset = load_dataset(args.task_config)
        task_names = list(dataset.keys())
        
        if args.max_tasks:
            task_names = task_names[:args.max_tasks]
            logger.info(f"  Limited to first {args.max_tasks} tasks for debugging")
        
        logger.info(f"  Loaded {len(task_names)} tasks: {task_names[:5]}{'...' if len(task_names)>5 else ''}")
        
        # Step 3: 配置ELO judge（固定使用高质量judge）
        logger.info("\n[3/4] Configuring ELO judges...")
        config_data = load_yaml(args.config_path)
        
        # 仅使用最可靠的judge（Claude Opus + Gemini Pro）
        primary_judges = ["claude-opus-4-5-20251101-thinking", "gemini-3-pro-preview"]
        api_configs = {}
        
        for judge in primary_judges:
            if judge in config_data['models']:
                conf = config_data['models'][judge]
                api_configs[judge] = {
                    "api_key": conf['api_key'],
                    "base_url": conf['base_url'],
                    "provider_prefix": "openai/"  # 假设均为OpenAI兼容
                }
            else:
                logger.warning(f"Judge {judge} not found in config, skipping")
        
        judge_config = {
            "primary": primary_judges,
            "reserves": [],
            "api_configs": api_configs
        }
        logger.info(f"  Using judges: {primary_judges}")
        
        # Step 4: 逐任务评估
        logger.info(f"\n[4/4] Evaluating {len(task_names)} tasks with 10 prompts each...")
        all_task_results = []
        
        for idx, task_name in enumerate(task_names, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"TASK {idx}/{len(task_names)}: {task_name}")
            logger.info(f"{'='*70}")
            
            try:
                task_result = evaluate_single_task(
                    task_name=task_name,
                    task_data=dataset[task_name],
                    prompts=prompts,
                    config_path=args.config_path,
                    judge_config=judge_config,
                    output_base_dir=output_base_dir
                )
                all_task_results.append(task_result)
                
                # 保存中间聚合结果（便于中断恢复）
                if idx % 5 == 0:
                    aggregate_results(all_task_results[:idx], output_base_dir)
                    
            except Exception as e:
                logger.exception(f"✗ Task {task_name} failed with error: {e}")
                continue
        
        # Step 5: 聚合最终结果并可视化
        logger.info("\n" + "="*70)
        logger.info("AGGREGATING FINAL RESULTS")
        logger.info("="*70)
        
        agg_result = aggregate_results(all_task_results, output_base_dir)
        
        # 保存完整结果
        final_result = {
            "metadata": {
                "prompt_count": 10,
                "task_count": len(all_task_results),
                "executor_model": "claude-opus-4-5-20251101-thinking",
                "evaluation_timestamp": datetime.now().isoformat(),
                "prompt_dir": args.prompt_dir
            },
            "task_results": all_task_results,
            "aggregated": agg_result
        }
        
        final_path = os.path.join(output_base_dir, "final_evaluation_results.json")
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Final results saved to {final_path}")
        logger.info(f"✓ Detailed logs per task in {os.path.join(output_base_dir, 'task_logs')}")
        logger.info(f"✓ Visualization in {os.path.join(output_base_dir, 'prompt_optimization_progress.png')}")
        logger.info("\n" + "="*70)
        logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":
    main()
