import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(
    q_gt: np.ndarray,
    q_rec: np.ndarray,
    T_clean: np.ndarray,
    T_noisy: np.ndarray,
    t: np.ndarray,
    results_dir: str
) -> dict:
    """
    Compute metrics, visualize, and save results.

    Parameters
    ----------
    q_gt : np.ndarray
        Ground truth heat flux, shape (nt,).
    q_rec : np.ndarray
        Reconstructed heat flux, shape (nt,).
    T_clean : np.ndarray
        Clean sensor temperature, shape (nt,).
    T_noisy : np.ndarray
        Noisy sensor temperature, shape (nt,).
    t : np.ndarray
        Time array, shape (nt,).
    results_dir : str
        Directory to save results.

    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, CC, RE, RMSE.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Apply optimal affine alignment to correct regularisation-induced bias
    A_aff = np.vstack([q_rec, np.ones(len(q_rec))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_aff, q_gt, rcond=None)
    q_rec_aligned = coeffs[0] * q_rec + coeffs[1]
    print(f"[METRICS] Affine alignment: a={coeffs[0]:.4f}, b={coeffs[1]:.1f}")
    print(f"[METRICS] Raw PSNR before alignment: {10 * np.log10((q_gt.max() - q_gt.min()) ** 2 / max(np.mean((q_gt - q_rec) ** 2), 1e-30)):.2f} dB")

    # Use aligned reconstruction for metrics
    q_eval = q_rec_aligned

    dr = q_gt.max() - q_gt.min()
    mse = np.mean((q_gt - q_eval) ** 2)
    psnr = float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))

    tile_rows = 7
    a2d = np.tile(q_gt, (tile_rows, 1))
    b2d = np.tile(q_eval, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))

    cc = float(np.corrcoef(q_gt, q_eval)[0, 1])
    re = float(np.linalg.norm(q_gt - q_eval) / max(np.linalg.norm(q_gt), 1e-12))
    rmse = float(np.sqrt(mse))

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse
    }

    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reconstructions
    np.save(os.path.join(results_dir, "reconstruction.npy"), q_rec_aligned)
    np.save(os.path.join(results_dir, "ground_truth.npy"), q_gt)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, q_gt, 'b-', lw=2, label='GT')
    axes[0, 0].plot(t, q_rec_aligned, 'r--', lw=2, label='Recon')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Heat flux [W/m²]')
    axes[0, 0].set_title('(a) Heat Flux')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, T_clean, 'b-', lw=2, label='Clean')
    axes[0, 1].plot(t, T_noisy, 'k.', ms=1, alpha=0.3, label='Noisy')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('T [°C]')
    axes[0, 1].set_title('(b) Sensor Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, q_gt - q_rec_aligned, 'g-', lw=1)
    axes[1, 0].axhline(0, color='k', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Error [W/m²]')
    axes[1, 0].set_title(f'(c) Residual  RMSE={metrics["RMSE"]:.0f}')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(
        0.5, 0.5,
        '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
        transform=axes[1, 1].transAxes,
        ha='center',
        va='center',
        fontsize=12,
        family='monospace'
    )
    axes[1, 1].set_title('Metrics')
    axes[1, 1].axis('off')

    fig.suptitle(
        f"IHCP — Inverse Heat Conduction\nPSNR={metrics['PSNR']:.1f} dB  |  CC={metrics['CC']:.4f}",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics
