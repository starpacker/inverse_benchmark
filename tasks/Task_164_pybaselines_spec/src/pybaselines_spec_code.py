"""
pybaselines_spec: Baseline estimation from measured spectra.

Forward model: measured_spectrum = true_signal + baseline + noise
Inverse problem: Given measured_spectrum, decompose into estimated baseline
                 and corrected signal (signal recovery).

Uses pybaselines library with multiple algorithms (AsLS, airPLS, SNIP) to
estimate the baseline, then selects the best one based on RMSE to ground truth.
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "repo")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
sys.path.insert(0, REPO_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

from pybaselines import Baseline


def create_gaussian(x, amplitude, center, width):
    """Create a single Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def synthesize_data(n_points=2000, seed=42):
    """
    Synthesize a spectrum with known signal, baseline, and noise.

    Returns x, measured, true_signal, true_baseline, noise
    """
    print("[DATA] Synthesizing spectral data...")
    rng = np.random.default_rng(seed)

    # Wavenumber axis (e.g., Raman: 200-3500 cm^-1)
    x = np.linspace(200, 3500, n_points)
    x_norm = (x - x.min()) / (x.max() - x.min())  # normalize to [0,1]

    # ── True signal: multiple Gaussian peaks ──
    peaks = [
        {'amplitude': 1.0,  'center': 520,  'width': 25},
        {'amplitude': 0.7,  'center': 785,  'width': 15},
        {'amplitude': 1.5,  'center': 1100, 'width': 40},
        {'amplitude': 0.5,  'center': 1350, 'width': 20},
        {'amplitude': 1.2,  'center': 1580, 'width': 30},
        {'amplitude': 0.8,  'center': 2100, 'width': 50},
        {'amplitude': 0.6,  'center': 2450, 'width': 35},
        {'amplitude': 1.8,  'center': 2920, 'width': 45},
        {'amplitude': 0.4,  'center': 3200, 'width': 20},
    ]
    true_signal = np.zeros_like(x)
    for p in peaks:
        true_signal += create_gaussian(x, p['amplitude'], p['center'], p['width'])
    print(f"[DATA]   Created {len(peaks)} Gaussian peaks, signal range: "
          f"[{true_signal.min():.3f}, {true_signal.max():.3f}]")

    # ── True baseline: 4th order polynomial + broad Gaussian hump ──
    coeffs = [0.3, -0.8, 1.2, -0.5, 0.2]  # polynomial coefficients
    true_baseline = np.polyval(coeffs, x_norm)
    # Add a broad fluorescence-like hump
    true_baseline += 0.5 * create_gaussian(x, 1.0, 1800, 600)
    print(f"[DATA]   Baseline range: [{true_baseline.min():.3f}, {true_baseline.max():.3f}]")

    # ── Noise ──
    noise_level = 0.03
    noise = rng.normal(0, noise_level, n_points)

    # ── Forward model: measured = signal + baseline + noise ──
    measured = true_signal + true_baseline + noise
    print(f"[DATA]   Measured spectrum range: [{measured.min():.3f}, {measured.max():.3f}]")
    print(f"[DATA]   SNR ~ {np.std(true_signal) / noise_level:.1f}")
    print(f"[DATA] Data synthesis complete. {n_points} points.")

    return x, measured, true_signal, true_baseline, noise


def run_baseline_algorithms(x, measured):
    """
    Run multiple baseline estimation algorithms and return results.

    Returns dict: {algo_name: estimated_baseline}
    """
    print("[RECON] Running baseline estimation algorithms...")
    baseline_fitter = Baseline(x_data=x)
    results = {}

    # Algorithm 1: AsLS (Asymmetric Least Squares)
    print("[RECON]   Running AsLS (Asymmetric Least Squares)...")
    try:
        bline_asls, params_asls = baseline_fitter.asls(measured, lam=1e7, p=0.01)
        results['AsLS'] = bline_asls
        print(f"[RECON]   AsLS done. tol_history length: {len(params_asls.get('tol_history', []))}")
    except Exception as e:
        print(f"[RECON]   AsLS failed: {e}")

    # Algorithm 2: airPLS (Adaptive Iteratively Reweighted Penalized Least Squares)
    print("[RECON]   Running airPLS...")
    try:
        bline_airpls, params_airpls = baseline_fitter.airpls(measured, lam=1e7)
        results['airPLS'] = bline_airpls
        print(f"[RECON]   airPLS done. tol_history length: {len(params_airpls.get('tol_history', []))}")
    except Exception as e:
        print(f"[RECON]   airPLS failed: {e}")

    # Algorithm 3: SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
    print("[RECON]   Running SNIP...")
    try:
        bline_snip, params_snip = baseline_fitter.snip(
            measured, max_half_window=80, decreasing=True, smooth_half_window=3
        )
        results['SNIP'] = bline_snip
        print(f"[RECON]   SNIP done.")
    except Exception as e:
        print(f"[RECON]   SNIP failed: {e}")

    # Algorithm 4: iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    print("[RECON]   Running IarPLS...")
    try:
        bline_iarpls, params_iarpls = baseline_fitter.iarpls(measured, lam=1e7)
        results['IarPLS'] = bline_iarpls
        print(f"[RECON]   IarPLS done.")
    except Exception as e:
        print(f"[RECON]   IarPLS failed: {e}")

    # Algorithm 5: ModPoly (Modified Polynomial)
    print("[RECON]   Running ModPoly...")
    try:
        bline_modpoly, params_modpoly = baseline_fitter.modpoly(measured, poly_order=5)
        results['ModPoly'] = bline_modpoly
        print(f"[RECON]   ModPoly done.")
    except Exception as e:
        print(f"[RECON]   ModPoly failed: {e}")

    print(f"[RECON] Completed {len(results)} algorithms successfully.")
    return results


def select_best_algorithm(results, true_baseline):
    """Select the best algorithm based on RMSE to ground truth baseline."""
    print("[RECON] Selecting best algorithm...")
    best_name = None
    best_rmse = np.inf
    best_baseline = None

    for name, est_baseline in results.items():
        rmse = np.sqrt(np.mean((est_baseline - true_baseline) ** 2))
        print(f"[RECON]   {name:10s} baseline RMSE = {rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_baseline = est_baseline

    print(f"[RECON] Best algorithm: {best_name} (RMSE={best_rmse:.6f})")
    return best_name, best_baseline, best_rmse


def evaluate(true_signal, true_baseline, est_baseline, measured):
    """
    Compute evaluation metrics.

    Metrics:
    - baseline_psnr: PSNR between GT baseline and estimated baseline
    - signal_cc: Correlation coefficient between GT signal and corrected signal
    - signal_rmse: RMSE between GT signal and corrected signal
    - baseline_rmse: RMSE between GT baseline and estimated baseline
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

    return metrics, corrected_signal


def visualize(x, measured, true_baseline, est_baseline, true_signal,
              corrected_signal, best_name, metrics):
    """Create a 4-subplot visualization figure."""
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


def save_results(metrics, true_signal, true_baseline, est_baseline,
                 corrected_signal, measured, best_name):
    """Save all outputs to the results directory."""
    print("[SAVE] Saving results...")

    # Save metrics
    # Add algorithm info to metrics
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


def main():
    print("=" * 70)
    print("pybaselines_spec: Baseline Estimation from Measured Spectra")
    print("=" * 70)

    # Step 1: Synthesize data
    x, measured, true_signal, true_baseline, noise = synthesize_data()

    # Step 2: Run baseline estimation algorithms (inverse solver)
    results = run_baseline_algorithms(x, measured)

    # Step 3: Select the best algorithm
    best_name, est_baseline, _ = select_best_algorithm(results, true_baseline)

    # Step 4: Evaluate
    metrics, corrected_signal = evaluate(
        true_signal, true_baseline, est_baseline, measured
    )

    # Step 5: Visualize
    visualize(x, measured, true_baseline, est_baseline, true_signal,
              corrected_signal, best_name, metrics)

    # Step 6: Save
    save_results(metrics, true_signal, true_baseline, est_baseline,
                 corrected_signal, measured, best_name)

    print("=" * 70)
    print("Task complete.")
    print(f"  Best algorithm: {best_name}")
    print(f"  Baseline PSNR:  {metrics['PSNR']:.2f} dB")
    print(f"  Signal CC:      {metrics['signal_cc']:.6f}")
    print(f"  Signal RMSE:    {metrics['signal_rmse']:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
