import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

def normalise_phase(phase):
    """Shift to min=0 and normalise to [0, 1]."""
    p = phase - phase.min()
    mx = p.max()
    return p / mx if mx > 0 else p

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

def evaluate_results(gt_phase, recon_phase, fov_mask, data_4d, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE metrics between ground-truth and reconstructed
    phase, handles phase alignment, and generates visualizations.
    
    Parameters
    ----------
    gt_phase : ndarray
        Ground truth phase array
    recon_phase : ndarray
        Reconstructed phase array
    fov_mask : ndarray (bool)
        Field-of-view mask indicating valid reconstruction region
    data_4d : ndarray
        Raw 4D-STEM data for visualization
    results_dir : str
        Directory to save results
    
    Returns
    -------
    dict containing:
        - PSNR_dB: Peak signal-to-noise ratio in dB
        - SSIM: Structural similarity index
        - RMSE: Root mean square error
        - phase_correlation: Correlation between GT and reconstruction
        - psnr, ssim, rmse: Duplicate keys for compatibility
    """
    from skimage.metrics import structural_similarity as ssim

    print("\n[6/6] Computing metrics ...")

    assert gt_phase.shape == recon_phase.shape, (
        f"Shape mismatch: GT {gt_phase.shape} vs recon {recon_phase.shape}"
    )

    fov = fov_mask
    gt_fov = gt_phase[fov]
    rc_fov = recon_phase[fov]

    # Remove global phase offset
    rc_fov = rc_fov - np.mean(rc_fov)
    gt_fov = gt_fov - np.mean(gt_fov)

    # Handle sign ambiguity
    corr_pos = np.corrcoef(gt_fov, rc_fov)[0, 1]
    corr_neg = np.corrcoef(gt_fov, -rc_fov)[0, 1]
    if corr_neg > corr_pos:
        recon_phase = -recon_phase
        rc_fov = -rc_fov
        print("  (Phase sign flipped for alignment)")

    best_corr = max(corr_pos, corr_neg)
    print(f"  Phase correlation (FOV) = {best_corr:.4f}")
    print(f"  FOV pixels              = {fov.sum()} / {fov.size}")

    # Alignment via least-squares
    recon_aligned = recon_phase.copy()
    gt_aligned = gt_phase.copy()

    gt_fov_vals = gt_aligned[fov].flatten()
    rc_fov_vals = recon_aligned[fov].flatten()
    A_mat = np.column_stack([gt_fov_vals, np.ones_like(gt_fov_vals)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, rc_fov_vals, rcond=None)
    a_ls, b_ls = coeffs
    print(f"  LS alignment: recon ≈ {a_ls:.4f} * GT + {b_ls:.6f}")
    
    if abs(a_ls) > 1e-10:
        recon_aligned = (recon_aligned - b_ls) / a_ls

    # FOV-only PSNR/RMSE
    gt_fov_pixels = gt_aligned[fov]
    rc_fov_pixels = recon_aligned[fov]
    rc_fov_pixels = np.clip(rc_fov_pixels, gt_fov_pixels.min(), gt_fov_pixels.max())
    gt_n_fov = (gt_fov_pixels - gt_fov_pixels.min()) / (gt_fov_pixels.max() - gt_fov_pixels.min() + 1e-12)
    rc_n_fov = (rc_fov_pixels - gt_fov_pixels.min()) / (gt_fov_pixels.max() - gt_fov_pixels.min() + 1e-12)
    rc_n_fov = np.clip(rc_n_fov, 0, 1)
    rmse_fov = float(np.sqrt(np.mean((gt_n_fov - rc_n_fov)**2)))
    psnr_fov = float(20.0 * np.log10(1.0 / rmse_fov)) if rmse_fov > 0 else float('inf')

    # FOV SSIM using bounding box
    rows = np.any(fov, axis=1)
    cols = np.any(fov, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    gt_box = gt_aligned[rmin:rmax + 1, cmin:cmax + 1].copy()
    rc_box = recon_aligned[rmin:rmax + 1, cmin:cmax + 1].copy()
    fov_box = fov[rmin:rmax + 1, cmin:cmax + 1]

    rc_box = np.clip(rc_box, gt_box[fov_box].min(), gt_box[fov_box].max())
    gt_box[~fov_box] = np.mean(gt_box[fov_box])
    rc_box[~fov_box] = np.mean(rc_box[fov_box])
    gt_rng = gt_box.max() - gt_box.min() + 1e-12
    gt_box_n = (gt_box - gt_box.min()) / gt_rng
    rc_box_n = (rc_box - gt_box.min()) / gt_rng
    rc_box_n = np.clip(rc_box_n, 0, 1)
    ssim_val = float(ssim(gt_box_n, rc_box_n, data_range=1.0))

    metrics = {
        "PSNR_dB": round(psnr_fov, 3),
        "SSIM": round(ssim_val, 4),
        "RMSE": round(rmse_fov, 6),
    }
    metrics["phase_correlation"] = round(float(best_corr), 4)
    metrics["psnr"] = metrics["PSNR_dB"]
    metrics["ssim"] = metrics["SSIM"]
    metrics["rmse"] = metrics["RMSE"]
    
    print(f"  PSNR = {metrics['psnr']:.2f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CC   = {metrics['phase_correlation']:.4f}")

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"\n  Metrics  → {metrics_path}")

    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_aligned)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_phase)
    print(f"  Arrays   → results/reconstruction.npy, ground_truth.npy")

    # Visualize
    avg_dp = data_4d.mean(axis=(0, 1))
    gt_n = normalise_phase(gt_aligned)
    rc_n = normalise_phase(recon_aligned)
    err = gt_n - rc_n

    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plot_results(gt_n, avg_dp, rc_n, err, metrics, fig_path)

    return metrics
