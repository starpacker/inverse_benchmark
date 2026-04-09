import matplotlib

matplotlib.use('Agg')

import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_results_internal(gt, recon, metrics, method_name, save_path):
    """
    Create 6-panel visualization.
    """
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    mid = gt.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    slice_labels = ['Axial (XY)', 'Coronal (XZ)', 'Sagittal (YZ)']
    gt_slices = [gt_norm[mid, :, :], gt_norm[:, mid, :], gt_norm[:, :, mid]]
    recon_slices = [recon_norm[mid, :, :], recon_norm[:, mid, :], recon_norm[:, :, mid]]

    for j in range(3):
        im0 = axes[0, j].imshow(gt_slices[j], cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f'GT {slice_labels[j]}', fontsize=12, fontweight='bold')
        axes[0, j].axis('off')
        plt.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)

        im1 = axes[1, j].imshow(recon_slices[j], cmap='gray', vmin=0, vmax=1)
        axes[1, j].set_title(f'Recon {slice_labels[j]}', fontsize=12, fontweight='bold')
        axes[1, j].axis('off')
        plt.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)

    metrics_text = (
        f"Method: {method_name}\n"
        f"PSNR: {metrics['psnr']:.2f} dB | CC: {metrics['cc']:.4f}\n"
        f"SSIM (avg): {metrics['ssim_avg']:.4f} | RMSE: {metrics['rmse']:.4f}"
    )
    fig.suptitle(
        f"Cryo-EM 3D Reconstruction: {method_name}\n{metrics_text}",
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {save_path}")
