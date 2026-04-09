import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import sys

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

def evaluate_results(ground_truth, reconstruction, results_dir):
    """
    Evaluate reconstruction quality by comparing to ground truth.
    
    Computes:
      - PSNR (Peak Signal-to-Noise Ratio) in dB
      - RMSE (Root Mean Square Error)
      - Correlation Coefficient (Pearson)
      - Relative Error
    
    Also generates visualization and saves metrics.
    
    Parameters:
        ground_truth: dict with 'frequency', 'n', 'k', 'alpha'
        reconstruction: dict with 'frequency', 'n', 'k', 'alpha'
        results_dir: directory to save results
    
    Returns:
        dict of metrics
    """
    
    def compute_psnr(ref, test, data_range=None):
        """Compute PSNR (dB) for 1D signals."""
        if data_range is None:
            data_range = ref.max() - ref.min()
        mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)

    def compute_rmse(ref, test):
        """Compute RMSE."""
        return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))

    def compute_correlation(ref, test):
        """Compute Pearson correlation coefficient."""
        r = np.corrcoef(ref.flatten(), test.flatten())[0, 1]
        return r

    def compute_relative_error(ref, test):
        """Compute relative error ||ref - test|| / ||ref||."""
        return np.linalg.norm(ref - test) / np.linalg.norm(ref)

    def interpolate_to_common_freq(gt_freq, gt_values, recon_freq, recon_values):
        """
        Interpolate reconstructed values onto ground truth frequency grid
        for fair comparison.
        """
        f_min = max(gt_freq.min(), recon_freq.min())
        f_max = min(gt_freq.max(), recon_freq.max())
        
        mask = (gt_freq >= f_min) & (gt_freq <= f_max)
        common_freq = gt_freq[mask]
        gt_common = gt_values[mask]
        
        recon_common = np.interp(common_freq, recon_freq, recon_values)
        
        return common_freq, gt_common, recon_common

    # Interpolate to common frequency grid
    # Refractive index n
    common_freq_n, gt_n, recon_n = interpolate_to_common_freq(
        ground_truth['frequency'], ground_truth['n'],
        reconstruction['frequency'], reconstruction['n']
    )
    
    # Extinction coefficient k
    common_freq_k, gt_k, recon_k = interpolate_to_common_freq(
        ground_truth['frequency'], ground_truth['k'],
        reconstruction['frequency'], reconstruction['k']
    )
    
    # Absorption alpha (convert to cm^-1)
    common_freq_a, gt_alpha, recon_alpha = interpolate_to_common_freq(
        ground_truth['frequency'], ground_truth['alpha'] * 0.01,
        reconstruction['frequency'], reconstruction['alpha'] * 0.01
    )

    # Compute metrics
    metrics = {
        # n metrics
        "psnr_n": float(compute_psnr(gt_n, recon_n)),
        "cc_n": float(compute_correlation(gt_n, recon_n)),
        "rmse_n": float(compute_rmse(gt_n, recon_n)),
        "re_n": float(compute_relative_error(gt_n, recon_n)),
        # k metrics
        "psnr_k": float(compute_psnr(gt_k, recon_k)),
        "cc_k": float(compute_correlation(gt_k, recon_k)),
        "rmse_k": float(compute_rmse(gt_k, recon_k)),
        # alpha metrics
        "cc_alpha": float(compute_correlation(gt_alpha, recon_alpha)),
        "rmse_alpha": float(compute_rmse(gt_alpha, recon_alpha)),
        # Overall (use n as primary)
        "psnr": float(compute_psnr(gt_n, recon_n)),
        "rmse": float(compute_rmse(gt_n, recon_n)),
    }

    # Print metrics
    print(f"[EVAL] n — PSNR = {metrics['psnr_n']:.4f} dB")
    print(f"[EVAL] n — CC   = {metrics['cc_n']:.8f}")
    print(f"[EVAL] n — RMSE = {metrics['rmse_n']:.8f}")
    print(f"[EVAL] n — RE   = {metrics['re_n']:.8f}")
    print(f"[EVAL] k — PSNR = {metrics['psnr_k']:.4f} dB")
    print(f"[EVAL] k — CC   = {metrics['cc_k']:.8f}")
    print(f"[EVAL] α — CC   = {metrics['cc_alpha']:.8f}")
    print(f"[EVAL] α — RMSE = {metrics['rmse_alpha']:.8f}")

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    gt_freq_THz = ground_truth['frequency'] / 1e12
    recon_freq_THz = reconstruction['frequency'] / 1e12
    
    # (a) Refractive index n(ω) comparison
    ax = axes[0, 0]
    ax.plot(gt_freq_THz, ground_truth['n'], 'b.', markersize=2, alpha=0.6, label='Ground Truth')
    ax.plot(recon_freq_THz, reconstruction['n'], 'r-', linewidth=1.0, label='Extraction')
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Refractive Index n')
    ax.set_title('Refractive Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Absorption coefficient α(ω) comparison
    ax = axes[0, 1]
    gt_alpha_cm = ground_truth['alpha'] * 0.01
    recon_alpha_cm = reconstruction['alpha'] * 0.01
    ax.plot(gt_freq_THz, gt_alpha_cm, 'b.', markersize=2, alpha=0.6, label='Ground Truth')
    ax.plot(recon_freq_THz, recon_alpha_cm, 'r-', linewidth=1.0, label='Extraction')
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel(r'Absorption $\alpha$ (cm$^{-1}$)')
    ax.set_title('Absorption Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Refractive index residual
    ax = axes[1, 0]
    residual_n = recon_n - gt_n
    ax.plot(common_freq_n / 1e12, residual_n, 'g-', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Δn (Extracted − GT)')
    ax.set_title(f'n Residual (RMSE={metrics["rmse_n"]:.6f})')
    ax.grid(True, alpha=0.3)
    
    # (d) Absorption residual
    ax = axes[1, 1]
    residual_a = recon_alpha - gt_alpha
    ax.plot(common_freq_a / 1e12, residual_a, 'm-', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel(r'Δα (cm$^{-1}$)')
    ax.set_title(f'α Residual (RMSE={metrics["rmse_alpha"]:.6f})')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"THz-TDS Parameter Extraction — PSNR_n={metrics['psnr_n']:.2f} dB | "
        f"CC_n={metrics['cc_n']:.6f} | "
        f"CC_α={metrics['cc_alpha']:.6f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")

    # Save arrays
    gt_array = np.column_stack([common_freq_n, gt_n, gt_k[:len(common_freq_n)]])
    recon_array = np.column_stack([common_freq_n, recon_n, recon_k[:len(common_freq_n)]])
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_array)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_array)
    
    measurement = np.column_stack([
        reconstruction['frequency'],
        reconstruction['n'],
        reconstruction['k'],
        reconstruction['alpha']
    ])
    np.save(os.path.join(results_dir, "input.npy"), measurement)

    return metrics
