import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_results(gt, zero_filled, cs_recon, metrics_zf, metrics_cs, save_path):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    gt_disp = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    zf_disp = (zero_filled - zero_filled.min()) / (zero_filled.max() - zero_filled.min() + 1e-12)
    cs_disp = (cs_recon - cs_recon.min()) / (cs_recon.max() - cs_recon.min() + 1e-12)
    error_map = np.abs(gt_disp - cs_disp)

    axes[0].imshow(gt_disp, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(zf_disp, cmap='gray')
    axes[1].set_title(f'Zero-filled\nPSNR={metrics_zf["PSNR"]:.2f} SSIM={metrics_zf["SSIM"]:.3f}',
                      fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(cs_disp, cmap='gray')
    axes[2].set_title(f'CS-TV Recon (ISTA)\nPSNR={metrics_cs["PSNR"]:.2f} SSIM={metrics_cs["SSIM"]:.3f}',
                      fontsize=11)
    axes[2].axis('off')

    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.15)
    axes[3].set_title(f'Error Map (CS)\nRMSE={metrics_cs["RMSE"]:.4f}', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('FastMRI Reconstruction: Accelerated MRI from Undersampled k-Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")
