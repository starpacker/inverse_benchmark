import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def plot_results(gt_phase, avg_dp, recon_phase, error_map, metrics, save_path):
    """4-panel figure: GT | avg DP | recon | error."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].imshow(gt_phase, cmap="inferno")
    axes[0].set_title("Ground-Truth Phase")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.log1p(avg_dp), cmap="viridis")
    axes[1].set_title("Avg Diffraction (log)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(recon_phase, cmap="inferno")
    axes[2].set_title(
        f"Reconstructed Phase\n"
        f"PSNR={metrics['PSNR_dB']:.1f} dB  SSIM={metrics['SSIM']:.3f}"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error_map, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[3].set_title(f"Phase Error (RMSE={metrics['RMSE']:.4f})")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")
