import matplotlib

matplotlib.use('Agg')

import os

import sys

import json

import numpy as np

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, "repo")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

sys.path.insert(0, REPO_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(true_signal, true_baseline, est_baseline, measured,
                     best_name, x):
    """
    Compute evaluation metrics, create visualization, and save results.
    
    Metrics computed:
    - baseline_psnr: PSNR between GT baseline and estimated baseline
    - signal_cc: Correlation coefficient between GT signal and corrected signal
    - signal_rmse: RMSE between GT signal and corrected signal
    - baseline_rmse: RMSE between GT baseline and estimated baseline
    
    Parameters
    ----------
    true_signal : np.ndarray
        Ground truth signal.
    true_baseline : np.ndarray
        Ground truth baseline.
    est_baseline : np.ndarray
        Estimated baseline from inversion.
    measured : np.ndarray
        Measured spectrum.
    best_name : str
        Name of the best algorithm used.
    x : np.ndarray
        Wavenumber axis for plotting.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    print("[EVAL] Computing evaluation metrics...")

    # Corrected signal = measured - estimated baseline
    corrected_signal = measured - est_baseline

    # Baseline PSNR
    baseline_mse = np.mean((true_baseline - est_baseline) ** 2)
    baseline_range = np.max(true_baseline) - np.min(true_baseline)
    if baseline_mse > 0:
        baseline_psnr = 10 * np.log10(baseline_range ** 2 / baseline_mse)
    else:
        baseline_psnr = float('inf')

    # Signal Correlation Coefficient
    signal_cc = np.corrcoef(true_signal, corrected_signal)[0, 1]

    # Signal RMSE
    signal_rmse = np.sqrt(np.mean((true_signal - corrected_signal) ** 2))

    # Baseline RMSE
    baseline_rmse = np.sqrt(baseline_mse)

    # Signal PSNR (main PSNR for report)
    signal_mse = np.mean((true_signal - corrected_signal) ** 2)
    signal_range = np.max(true_signal) - np.min(true_signal)
    if signal_mse > 0:
        signal_psnr = 10 * np.log10(signal_range ** 2 / signal_mse)
    else:
        signal_psnr = float('inf')

    metrics = {
        'PSNR': float(round(baseline_psnr, 4)),
        'signal_psnr': float(round(signal_psnr, 4)),
        'signal_cc': float(round(signal_cc, 6)),
        'signal_rmse': float(round(signal_rmse, 6)),
        'baseline_rmse': float(round(baseline_rmse, 6)),
    }

    print(f"[EVAL]   Baseline PSNR:   {metrics['PSNR']:.2f} dB")
    print(f"[EVAL]   Signal PSNR:     {metrics['signal_psnr']:.2f} dB")
    print(f"[EVAL]   Signal CC:       {metrics['signal_cc']:.6f}")
    print(f"[EVAL]   Signal RMSE:     {metrics['signal_rmse']:.6f}")
    print(f"[EVAL]   Baseline RMSE:   {metrics['baseline_rmse']:.6f}")
    print("[EVAL] Evaluation complete.")

    # ── Visualization ──
    print("[VIS] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Baseline Estimation via pybaselines ({best_name})\n'
                 f'Baseline PSNR={metrics["PSNR"]:.1f} dB | '
                 f'Signal CC={metrics["signal_cc"]:.4f} | '
                 f'Signal RMSE={metrics["signal_rmse"]:.4f}',
                 fontsize=13, fontweight='bold')

    # (a) Measured spectrum with true baseline overlaid
    ax = axes[0, 0]
    ax.plot(x, measured, 'b-', alpha=0.6, linewidth=0.5, label='Measured')
    ax.plot(x, true_baseline, 'r-', linewidth=2, label='True Baseline')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('(a) Measured Spectrum + True Baseline')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Estimated baseline vs true baseline
    ax = axes[0, 1]
    ax.plot(x, true_baseline, 'r-', linewidth=2, label='True Baseline')
    ax.plot(x, est_baseline, 'g--', linewidth=2, label=f'Estimated ({best_name})')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'(b) Baseline Comparison (PSNR={metrics["PSNR"]:.1f} dB)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Corrected signal vs true signal
    ax = axes[1, 0]
    ax.plot(x, true_signal, 'r-', linewidth=1.5, label='True Signal')
    ax.plot(x, corrected_signal, 'b-', alpha=0.7, linewidth=0.8, label='Corrected Signal')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'(c) Signal Recovery (CC={metrics["signal_cc"]:.4f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Error (residual)
    ax = axes[1, 1]
    baseline_error = est_baseline - true_baseline
    signal_error = corrected_signal - true_signal
    ax.plot(x, baseline_error, 'g-', alpha=0.7, linewidth=0.8, label='Baseline Error')
    ax.plot(x, signal_error, 'b-', alpha=0.5, linewidth=0.5, label='Signal Error')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Error')
    ax.set_title(f'(d) Residual Errors (RMSE={metrics["signal_rmse"]:.4f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Figure saved to {fig_path}")

    # ── Save results ──
    print("[SAVE] Saving results...")

    # Save metrics
    metrics_out = dict(metrics)
    metrics_out['best_algorithm'] = best_name
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[SAVE]   metrics.json saved ({len(metrics_out)} entries)")

    # Save ground truth (signal + baseline stacked)
    gt = np.stack([true_signal, true_baseline], axis=0)
    gt_path = os.path.join(RESULTS_DIR, 'ground_truth.npy')
    np.save(gt_path, gt)
    print(f"[SAVE]   ground_truth.npy saved, shape={gt.shape}")

    # Save reconstruction (corrected signal + estimated baseline stacked)
    recon = np.stack([corrected_signal, est_baseline], axis=0)
    recon_path = os.path.join(RESULTS_DIR, 'reconstruction.npy')
    np.save(recon_path, recon)
    print(f"[SAVE]   reconstruction.npy saved, shape={recon.shape}")

    print("[SAVE] All results saved.")

    return metrics
