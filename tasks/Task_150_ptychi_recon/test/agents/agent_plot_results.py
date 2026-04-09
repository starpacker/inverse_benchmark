import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_results(gt, recon, metrics, errors, path):
    """Generate visualization of reconstruction results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    im0 = axes[0, 0].imshow(np.abs(gt), cmap='gray')
    axes[0, 0].set_title('GT Amplitude', fontsize=14)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.angle(gt), cmap='twilight',
                              vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('GT Phase', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].semilogy(errors, 'b-', lw=1.5)
    axes[0, 2].set(xlabel='Iteration', ylabel='Error',
                    title='ePIE Convergence')
    axes[0, 2].grid(True, alpha=0.3)

    im2 = axes[1, 0].imshow(np.abs(recon), cmap='gray')
    axes[1, 0].set_title(
        f'Recon Amplitude\nPSNR={metrics["psnr"]:.2f} dB  '
        f'SSIM={metrics["ssim"]:.4f}', fontsize=14)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(np.angle(recon), cmap='twilight',
                              vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title(
        f'Recon Phase\nPhase corr={metrics["phase_correlation"]:.4f}',
        fontsize=14)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    amp_err = np.abs(np.abs(gt) - np.abs(recon))
    im4 = axes[1, 2].imshow(amp_err, cmap='hot')
    axes[1, 2].set_title(f'Amplitude Error\nRMSE={metrics["rmse"]:.4f}',
                          fontsize=14)
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        if ax is not axes[0, 2]:
            ax.axis('off')

    plt.suptitle('Ptychographic Reconstruction (ePIE)\n'
                 f'Complex corr: {metrics["complex_correlation"]:.4f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot -> {path}")
