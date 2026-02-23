import matplotlib.pyplot as plt
import numpy as np

# 数据准备
epochs = ['epoch_01', 'epoch_02', 'epoch_03', 'epoch_04', 'epoch_05', 'epoch_06']
mean_values = [983.9, 965.2, 1013.4, 1029.7, 974.1, 967.9]
golden_value = 1065.8

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bars = ax.bar(epochs, mean_values, color='#4A90E2', edgecolor='black', alpha=0.85, width=0.6)

# 添加 Golden 水平参考线
ax.axhline(y=golden_value, color='red', linestyle='--', linewidth=2.5, label=f'Golden Plan: {golden_value:.1f}')

# 在柱子顶部标注数值
for bar, val in zip(bars, mean_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 8,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 图表美化
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Value', fontsize=13, fontweight='bold')
ax.set_title('Epoch Performance vs Golden Plan (Average)', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(900, 1100)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(fontsize=12, loc='upper right')
ax.set_axisbelow(True)  # 网格线置于底层

# 添加与 Golden 的差距标注
for i, (epoch, val) in enumerate(zip(epochs, mean_values)):
    gap = golden_value - val
    ax.text(i, golden_value + 10, f'-{gap:.1f}', ha='center', fontsize=9, color='darkred', alpha=0.7)

# 优化布局并保存
plt.tight_layout()
plt.savefig('./epoch_vs_golden.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存为 'epoch_vs_golden.png'")
plt.show()