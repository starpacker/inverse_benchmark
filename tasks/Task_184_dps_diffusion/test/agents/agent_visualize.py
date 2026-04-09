import matplotlib

matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt

def visualize(gt, degraded, recon, metrics, save_path):
    """4-panel figure: GT | Degraded | Reconstruction | Error map."""
    error = np.abs(gt - recon)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ['Ground Truth', 'Degraded (Blur+Noise)',
              f'DPS Reconstruction\nPSNR={metrics["psnr_db"]:.2f} dB  '
              f'SSIM={metrics["ssim"]:.4f}',
              'Absolute Error']

    for ax, img, title in zip(axes, [gt, degraded, recon, error], titles):
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.colorbar(axes[3].images[0], ax=axes[3], fraction=0.046)

    plt.suptitle('Diffusion Posterior Sampling (DPS) — Image Deblurring',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Saved visualization to {save_path}")
