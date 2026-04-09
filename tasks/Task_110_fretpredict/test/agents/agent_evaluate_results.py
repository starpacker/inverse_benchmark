import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from scipy.ndimage import gaussian_filter1d

def compute_psnr(gt, recon):
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(gt)
    if peak < 1e-12:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))

def compute_ssim_1d(gt, recon):
    """Compute a 1-D analogue of SSIM."""
    data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-12:
        return 0.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win_sigma = 11.0 / 6.0

    mu_x = gaussian_filter1d(gt, sigma=win_sigma)
    mu_y = gaussian_filter1d(recon, sigma=win_sigma)
    sig_x2 = gaussian_filter1d(gt ** 2, sigma=win_sigma) - mu_x ** 2
    sig_y2 = gaussian_filter1d(recon ** 2, sigma=win_sigma) - mu_y ** 2
    sig_xy = gaussian_filter1d(gt * recon, sigma=win_sigma) - mu_x * mu_y

    sig_x2 = np.maximum(sig_x2, 0)
    sig_y2 = np.maximum(sig_y2, 0)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2))
    return float(np.mean(ssim_map))

def compute_cc(gt, recon):
    g = gt - np.mean(gt)
    r = recon - np.mean(recon)
    denom = np.sqrt(np.sum(g ** 2) * np.sum(r ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(g * r) / denom)

def compute_rmse(gt, recon):
    return float(np.sqrt(np.mean((gt - recon) ** 2)))

def evaluate_results(p_gt, p_recon, r_grid, h, e_edges, params, results_dir, working_dir):
    """
    Compute metrics, save results, and generate visualization.
    
    Args:
        p_gt: ground truth distribution
        p_recon: reconstructed distribution
        r_grid: distance grid
        h: efficiency histogram
        e_edges: efficiency bin edges
        params: dictionary of parameters
        results_dir: directory to save results
        working_dir: working directory
        
    Returns:
        metrics: dictionary of computed metrics
    """
    R0 = params['R0']
    r_max = params['r_max']
    n_samples = params['n_samples']
    
    # Compute metrics
    psnr_val = compute_psnr(p_gt, p_recon)
    ssim_val = compute_ssim_1d(p_gt, p_recon)
    cc_val = compute_cc(p_gt, p_recon)
    rmse_val = compute_rmse(p_gt, p_recon)

    print(f"\n{'=' * 40}")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  CC:   {cc_val:.4f}")
    print(f"  RMSE: {rmse_val:.6f}")
    print(f"{'=' * 40}")

    metrics = {
        "PSNR": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "CC": round(cc_val, 4),
        "RMSE": round(rmse_val, 6),
    }

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), p_gt)
    np.save(os.path.join(results_dir, "reconstruction.npy"), p_recon)
    np.save(os.path.join(working_dir, "gt_output.npy"), p_gt)
    np.save(os.path.join(working_dir, "recon_output.npy"), p_recon)

    # Visualization
    e_centres = 0.5 * (e_edges[:-1] + e_edges[1:])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: True p(r)
    ax = axes[0, 0]
    ax.fill_between(r_grid, p_gt, alpha=0.4, color='steelblue')
    ax.plot(r_grid, p_gt, 'b-', linewidth=2)
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title("True Distance Distribution p(r)", fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    # Panel 2: FRET efficiency histogram
    ax = axes[0, 1]
    ax.bar(e_centres, h, width=e_centres[1] - e_centres[0],
           color='orange', alpha=0.7, edgecolor='darkorange')
    ax.set_xlabel("FRET Efficiency E", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Observed FRET Efficiency Histogram\n(N={n_samples} molecules)",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Panel 3: Recovered p(r)
    ax = axes[1, 0]
    ax.fill_between(r_grid, p_recon, alpha=0.4, color='tomato')
    ax.plot(r_grid, p_recon, 'r-', linewidth=2)
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title("Recovered Distance Distribution", fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    # Panel 4: Overlay comparison
    ax = axes[1, 1]
    ax.plot(r_grid, p_gt, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(r_grid, p_recon, 'r--', linewidth=2, label='Recovered')
    ax.fill_between(r_grid, p_gt, alpha=0.15, color='blue')
    ax.fill_between(r_grid, p_recon, alpha=0.15, color='red')
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title(f"Overlay Comparison\nPSNR={psnr_val:.1f}dB | SSIM={ssim_val:.4f} | CC={cc_val:.4f}",
                 fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    plt.suptitle("FRET Distance Distribution Recovery (Task 120)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to {results_dir}/")
    
    return metrics
