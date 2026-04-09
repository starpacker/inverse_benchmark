import numpy as np
import matplotlib.pyplot as plt

# ========== 修改这里切换样本 ==========
idx = 0  # 仅当数据为三维/四维（含样本维度）时生效；二维数据自动忽略此参数
# ====================================

# 检查依赖
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    has_skimage = True
except ImportError:
    has_skimage = False
    print("⚠️ 建议安装: pip install scikit-image")

# Min-Max归一化（支持任意维度）
def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# 标准化为二维单通道图像
def ensure_2d_grayscale(img):
    """
    将图像标准化为二维单通道格式
    - (H, W)     -> 保持不变
    - (H, W, 1)  -> 压缩为 (H, W)
    - (H, W, C)  -> 灰度转换（仅用于Input可视化）
    """
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        if img.shape[-1] == 1:
            return img[:, :, 0]
        elif img.shape[-1] in [3, 4]:  # RGB/RGBA
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            return img[:, :, 0]  # 非标准通道数，取第一通道
    else:
        raise ValueError(f"不支持的图像维度: {img.shape}")

# 安全索引：仅当数组有样本维度时才切片
def safe_index(arr, index):
    """
    智能索引：
    - 2D (H, W)       -> 直接返回（整个数组即一个样本）
    - 3D (N, H, W)    -> arr[index]
    - 4D (N, H, W, C) -> arr[index]
    """
    if arr.ndim == 2:
        print(f"  → 数据为2D ({arr.shape})，忽略 idx，直接使用整个数组作为样本")
        return arr
    elif arr.ndim >= 3:
        if index >= arr.shape[0]:
            raise IndexError(f"索引 {index} 超出范围（样本数: {arr.shape[0]}）")
        print(f"  → 数据为{arr.ndim}D ({arr.shape})，取第 {index} 个样本")
        N, H, W = arr.shape[:3]
        if H == W:
            return arr[index]
        if N == H:
            return arr[:,:, index]
        return arr[:,:,index] if N > W else arr[index]
    else:
        raise ValueError(f"不支持的数组维度: {arr.ndim}")

# ==================== 加载数据 ====================
input_data = np.load('dataset/input.npy')
baseline = np.load('dataset/baseline.npy')
output = np.load('output.npy')
gt = np.load('dataset/gt_output.npy')

print(f"原始数据 shapes:")
print(f"  Input:    {input_data.shape}")
print(f"  Baseline: {baseline.shape}")
print(f"  Output:   {output.shape}")
print(f"  GT:       {gt.shape}")

# ==================== 智能索引样本 ====================
print(f"\n正在提取样本 (idx={idx}):")
input_sample = safe_index(input_data, idx)
base_sample = safe_index(baseline, idx)
out_sample = safe_index(output, idx)
gt_sample = safe_index(gt, idx)

# ==================== 标准化为2D单通道 ====================
input_2d = ensure_2d_grayscale(input_sample)
base_2d = ensure_2d_grayscale(base_sample)
out_2d = ensure_2d_grayscale(out_sample)
gt_2d = ensure_2d_grayscale(gt_sample)

print(f"\n处理后样本 shapes (2D):")
print(f"  Input: {input_2d.shape} | Baseline: {base_2d.shape} | Output: {out_2d.shape} | GT: {gt_2d.shape}")

# ==================== 归一化用于评估 ====================
base_disp = minmax_norm(base_2d)
out_disp = minmax_norm(out_2d)
gt_disp = minmax_norm(gt_2d)

# ==================== 计算指标 ====================
base_mae = np.abs(base_disp - gt_disp).mean()
out_mae = np.abs(out_disp - gt_disp).mean()

if has_skimage:
    base_psnr = psnr(base_disp, gt_disp, data_range=1.0)
    out_psnr = psnr(out_disp, gt_disp, data_range=1.0)
    base_ssim = ssim(base_disp, gt_disp, data_range=1.0, channel_axis=None)
    out_ssim = ssim(out_disp, gt_disp, data_range=1.0, channel_axis=None)
else:
    def calc_psnr(x, y):
        mse = np.mean((x - y) ** 2)
        return 10 * np.log10(1.0 / (mse + 1e-10))
    base_psnr = calc_psnr(base_disp, gt_disp)
    out_psnr = calc_psnr(out_disp, gt_disp)
    base_ssim = out_ssim = np.nan


print("\n🔍 数据真实性验证:")
print(f"Output 与 GT 是否完全相等: {np.array_equal(output[idx], gt[idx])}")
print(f"Output 与 GT 的最大绝对差: {np.abs(output[idx] - gt[idx]).max():.2e}")
print(f"Output 数据范围: [{output[idx].min():.4f}, {output[idx].max():.4f}]")
print(f"GT 数据范围:     [{gt[idx].min():.4f}, {gt[idx].max():.4f}]")

if np.array_equal(output[idx], gt[idx]):
    raise RuntimeError(
        "❌ 致命错误: Output 与 GT 完全相同！\n"
        "   请检查:\n"
        "   1. output.npy 是否误保存了 gt_output.npy\n"
        "   2. 模型推理代码是否实际运行（而非直接复制GT）\n"
        "   3. 数据预处理流程是否存在逻辑错误"
    )

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 第一行：Input, GT, Baseline, Output
axes[0, 0].imshow(input_2d, cmap='viridis')
axes[0, 0].set_title('Input', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_2d, cmap='viridis')
axes[0, 1].set_title('GT', fontsize=12, fontweight='bold', color='green')
axes[0, 1].axis('off')

axes[0, 2].imshow(base_disp, cmap='viridis', vmin=0, vmax=1)
axes[0, 2].set_title(f'Baseline\nPSNR: {base_psnr:.1f}dB\nMAE: {base_mae:.4f}', 
                    fontsize=10, color='red')
axes[0, 2].axis('off')

axes[0, 3].imshow(out_disp, cmap='viridis', vmin=0, vmax=1)
axes[0, 3].set_title(f'Output\nPSNR: {out_psnr:.1f}dB\nMAE: {out_mae:.4f}', 
                    fontsize=10, color='blue')
axes[0, 3].axis('off')

# 第二行：误差图
axes[1, 0].axis('off')
axes[1, 1].axis('off')

base_error = np.abs(base_disp - gt_disp)
im1 = axes[1, 2].imshow(base_error, cmap='hot', vmin=0, vmax=base_error.max())
axes[1, 2].set_title(f'Baseline Error\nMAE: {base_mae:.4f}', fontsize=10, color='red')
axes[1, 2].axis('off')
plt.colorbar(im1, ax=axes[1, 2], fraction=0.046, pad=0.04)

out_error = np.abs(out_disp - gt_disp)
im2 = axes[1, 3].imshow(out_error, cmap='hot', vmin=0, vmax=out_error.max())
axes[1, 3].set_title(f'Output Error\nMAE: {out_mae:.4f}', fontsize=10, color='blue')
axes[1, 3].axis('off')
plt.colorbar(im2, ax=axes[1, 3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f'error_comparison_idx{idx}.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 打印指标 ====================
print(f"\n📊 样本指标 (min-max归一化后):")
print(f"Baseline - PSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f} | MAE: {base_mae:.4f}")
print(f"Ours     - PSNR: {out_psnr:.2f} dB | SSIM: {out_ssim:.4f} | MAE: {out_mae:.4f}")
if not np.isnan(base_ssim):
    print(f"↑ PSNR: {out_psnr - base_psnr:+.2f} dB | SSIM: {out_ssim - base_ssim:+.4f} | MAE: {base_mae - out_mae:+.4f}")
else:
    print(f"↑ PSNR: {out_psnr - base_psnr:+.2f} dB | MAE: {base_mae - out_mae:+.4f}")