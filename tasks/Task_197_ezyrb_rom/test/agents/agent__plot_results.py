import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def _plot_results(gt_2d, pred_2d, error_2d, metrics, k_test, save_path):
    """
    Create a 4-panel figure:
      (a) Ground truth field
      (b) ROM prediction
      (c) Absolute error map
      (d) Metrics summary
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    vmin = min(gt_2d.min(), pred_2d.min())
    vmax = max(gt_2d.max(), pred_2d.max())

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_2d.T, origin='lower', cmap='hot', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax1.set_title(f'(a) Ground Truth (k = {k_test:.2f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Temperature', shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_2d.T, origin='lower', cmap='hot', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax2.set_title(f'(b) ROM Prediction (POD + RBF)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Temperature', shrink=0.8)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(error_2d.T, origin='lower', cmap='RdBu_r', aspect='equal',
                     extent=[0, 1, 0, 1])
    ax3.set_title('(c) Absolute Error', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='|GT - Prediction|', shrink=0.8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    metrics_text = (
        f"Reconstruction Metrics\n"
        f"{'='*35}\n\n"
        f"Test parameter:  k = {k_test:.3f}\n\n"
        f"PSNR:            {metrics['psnr']:.2f} dB\n"
        f"SSIM:            {metrics['ssim']:.4f}\n"
        f"RMSE:            {metrics['rmse']:.6f}\n"
        f"Relative L2:     {metrics['relative_l2']:.6f}\n\n"
        f"Training snapshots:  {metrics['n_train']}\n"
        f"POD modes used:      {metrics['n_pod_modes']}\n"
        f"Grid resolution:     {metrics['grid_size']}\n"
        f"Interpolation:       RBF\n"
    )
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=13, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))
    ax4.set_title('(d) Evaluation Metrics', fontsize=14, fontweight='bold')

    plt.suptitle('EZyRB: Reduced-Order Model for 2D Heat Conduction\n'
                 'Inverse Problem: Reconstruct temperature field from sparse parameter snapshots',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved to {save_path}")
