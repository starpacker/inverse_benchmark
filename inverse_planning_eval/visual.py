#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELO Ratings Multi-Task Analysis for Inverse Planning Evaluation
针对 /home/yjh/inverse_planning_eval/eval_results_similarity.json 的专业分析
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和可视化风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_elo_data(filepath):
    """加载JSON文件并提取ELO评分数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据结构
    if 'elo_ratings' not in data:
        raise KeyError("JSON中缺少 'elo_ratings' 字段")
    
    return data['elo_ratings']


def extract_model_ratings(elo_data):
    """
    从多任务ELO数据中提取评分矩阵
    
    Returns:
        rating_df: DataFrame (models × tasks)
        task_list: 任务列表
        model_list: 模型列表（排除Golden）
    """
    task_ratings = {}
    all_models = set()
    
    # 遍历所有任务
    for task_name, task_data in elo_data.items():
        if 'final_ratings' not in task_data:
            continue
            
        ratings = task_data['final_ratings']
        # 排除Golden参考模型（仅用于ELO计算基准）
        task_ratings[task_name] = {k: v for k, v in ratings.items() if k != 'Golden'}
        all_models.update(task_ratings[task_name].keys())
    
    # 构建评分矩阵
    models = sorted(all_models)
    tasks = sorted(task_ratings.keys())
    
    rating_matrix = []
    for model in models:
        row = []
        for task in tasks:
            rating = task_ratings[task].get(model, np.nan)
            row.append(rating)
        rating_matrix.append(row)
    
    df = pd.DataFrame(rating_matrix, index=models, columns=tasks)
    return df, tasks, models


def compute_statistics(rating_df):
    """计算各模型的统计指标（排除Golden）"""
    stats = pd.DataFrame({
        'mean': rating_df.mean(axis=1),
        'std': rating_df.std(axis=1),
        'min': rating_df.min(axis=1),
        'max': rating_df.max(axis=1),
        'median': rating_df.median(axis=1),
        'task_count': rating_df.count(axis=1),
        'range': rating_df.max(axis=1) - rating_df.min(axis=1)
    })
    
    # 按平均分降序排序
    stats = stats.sort_values('mean', ascending=False)
    return stats


def plot_comprehensive_analysis(rating_df, stats_df, output_dir):
    """生成完整的可视化分析报告"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = stats_df.index.tolist()
    means = stats_df['mean'].values
    stds = stats_df['std'].values
    
    # ========== 图1: 平均ELO评分柱状图 ==========
    plt.figure(figsize=(14, 7))
    x_pos = np.arange(len(models))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    bars = plt.bar(x_pos, means, yerr=stds, capsize=6, color=colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(x_pos, models, rotation=30, ha='right', fontsize=11)
    plt.ylabel('ELO Rating', fontsize=13, fontweight='bold')
    plt.title('Model Average ELO Ratings Across Tasks (w/ Std Dev)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.ylim(980, 1020)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'elo_mean_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: 任务×模型热力图 ==========
    plt.figure(figsize=(16, 10))
    sns.heatmap(rating_df.loc[models].T, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'ELO Rating'}, linewidths=0.5, linecolor='gray')
    plt.title('ELO Ratings Heatmap: Tasks vs Models', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=13, fontweight='bold')
    plt.ylabel('Tasks', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'elo_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: 模型评分分布箱线图 ==========
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=rating_df.loc[models].T, order=models, palette='Set2', width=0.7)
    sns.stripplot(data=rating_df.loc[models].T, order=models, color='black', 
                  alpha=0.4, size=4, jitter=0.2)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.ylabel('ELO Rating', fontsize=13, fontweight='bold')
    plt.title('ELO Rating Distribution per Model (All Tasks)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'elo_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图4: 顶级模型雷达图 ==========
    top_n = min(6, len(models))
    top_models = models[:top_n]
    
    # 标准化任务名称
    tasks = rating_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    for i, model in enumerate(top_models):
        values = rating_df.loc[model].values.tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, markersize=8)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, size=10)
    ax.set_ylim(970, 1030)
    ax.set_title('Top Models: Multi-Task ELO Performance', 
                 fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'elo_radar_top6.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图5: 任务难度分析（各任务平均ELO） ==========
    task_stats = pd.DataFrame({
        'mean': rating_df.mean(axis=0),
        'std': rating_df.std(axis=0),
        'model_count': rating_df.count(axis=0)
    }).sort_values('mean', ascending=True)
    
    plt.figure(figsize=(14, 8))
    y_pos = np.arange(len(task_stats))
    plt.barh(y_pos, task_stats['mean'], xerr=task_stats['std'], capsize=5,
             color=plt.cm.plasma(np.linspace(0.2, 0.8, len(task_stats))),
             edgecolor='black', alpha=0.85)
    
    for i, (task, row) in enumerate(task_stats.iterrows()):
        plt.text(row['mean'] + 1.5, i, f"{row['mean']:.1f}±{row['std']:.1f}", 
                va='center', fontsize=9, fontweight='bold')
    
    plt.yticks(y_pos, task_stats.index, fontsize=10)
    plt.xlabel('Average ELO Rating', fontsize=13, fontweight='bold')
    plt.title('Task Difficulty Analysis (Lower = Harder Task)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'task_difficulty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 所有可视化图表已保存至: {output_path.absolute()}")


def export_results(rating_df, stats_df, output_dir):
    """导出详细统计结果"""
    output_path = Path(output_dir)
    
    # 1. 原始评分矩阵
    rating_df.to_csv(output_path / 'elo_ratings_matrix.csv')
    
    # 2. 统计摘要（按平均分排序）
    stats_df.to_csv(output_path / 'elo_statistics_summary.csv')
    
    # 3. 详细排名报告
    report = stats_df.reset_index()
    report.columns = ['Model'] + list(stats_df.columns)
    report.insert(1, 'Rank', range(1, len(report) + 1))
    report = report[['Rank', 'Model', 'mean', 'std', 'min', 'max', 'median', 'range', 'task_count']]
    report.columns = ['Rank', 'Model', 'Mean ELO', 'Std Dev', 'Min', 'Max', 'Median', 'Range', 'Task Count']
    report.to_csv(output_path / 'elo_detailed_ranking.csv', index=False)
    
    # 4. Excel综合报告
    with pd.ExcelWriter(output_path / 'elo_comprehensive_report.xlsx') as writer:
        rating_df.to_excel(writer, sheet_name='Raw Ratings')
        stats_df.to_excel(writer, sheet_name='Model Statistics')
        report.to_excel(writer, sheet_name='Ranking Report', index=False)
        
        # 添加任务统计
        task_stats = pd.DataFrame({
            'Mean ELO': rating_df.mean(axis=0),
            'Std Dev': rating_df.std(axis=0),
            'Model Count': rating_df.count(axis=0)
        })
        task_stats.to_excel(writer, sheet_name='Task Statistics')
    
    print(f"✓ 详细统计结果已导出至: {output_path.absolute()}")


def generate_summary_report(stats_df, task_count, output_dir):
    """生成文本摘要报告"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ELO RATINGS MULTI-TASK ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTotal Tasks Analyzed: {task_count}")
    report_lines.append(f"Total Models Evaluated: {len(stats_df)}")
    report_lines.append("\n" + "-" * 70)
    report_lines.append("TOP 5 MODELS BY AVERAGE ELO RATING")
    report_lines.append("-" * 70)
    
    for i, (model, row) in enumerate(stats_df.head(5).iterrows(), 1):
        report_lines.append(
            f"{i}. {model:35s} | Mean: {row['mean']:6.2f} | Std: {row['std']:5.2f} | "
            f"Range: [{row['min']:6.2f}, {row['max']:6.2f}] | Tasks: {int(row['task_count'])}"
        )
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("MODEL STABILITY ANALYSIS (Low Std = Consistent Performance)")
    report_lines.append("-" * 70)
    
    stability_df = stats_df.sort_values('std').head(5)
    for i, (model, row) in enumerate(stability_df.iterrows(), 1):
        report_lines.append(
            f"{i}. {model:35s} | Std: {row['std']:5.2f} | Mean: {row['mean']:6.2f}"
        )
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append(f"Full results saved to: {output_dir}")
    report_lines.append("=" * 70)
    
    # 保存报告
    report_path = Path(output_dir) / 'analysis_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印到控制台
    print('\n'.join(report_lines))


def main():
    # 配置路径
    json_file = '/home/yjh/inverse_planning_eval/eval_results_similarity.json'
    output_dir = '/home/yjh/inverse_planning_eval/elo_analysis_results'
    
    print("\n" + "=" * 70)
    print("ELO MULTI-TASK PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/5] Loading ELO ratings data...")
    try:
        elo_data = load_elo_data(json_file)
        print(f"✓ Successfully loaded data from: {json_file}")
        print(f"  Found {len(elo_data)} tasks with ELO ratings")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 2. 提取评分
    print("\n[2/5] Extracting model ratings across tasks...")
    rating_df, tasks, models = extract_model_ratings(elo_data)
    print(f"✓ Extracted ratings for {len(models)} models across {len(tasks)} tasks")
    print(f"  Models: {', '.join(models[:4])}{'...' if len(models) > 4 else ''}")
    print(f"  Tasks: {', '.join(tasks[:4])}{'...' if len(tasks) > 4 else ''}")
    
    # 3. 计算统计量
    print("\n[3/5] Computing statistics...")
    stats_df = compute_statistics(rating_df)
    print("\nModel Rankings (by mean ELO):")
    print(stats_df[['mean', 'std', 'min', 'max', 'task_count']].round(2).head(10))
    
    # 4. 可视化
    print("\n[4/5] Generating visualizations...")
    plot_comprehensive_analysis(rating_df, stats_df, output_dir)
    
    # 5. 导出结果
    print("\n[5/5] Exporting results...")
    export_results(rating_df, stats_df, output_dir)
    
    # 6. 生成摘要报告
    generate_summary_report(stats_df, len(tasks), output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()