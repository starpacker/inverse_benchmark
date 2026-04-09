#!/usr/bin/env python3
"""
Task 176: mne_nirs_flim
fNIRS hemodynamic response recovery using Modified Beer-Lambert Law (MBLL).

Forward: Known Δ[HbO], Δ[HbR] → compute ΔOD at 760nm, 850nm
Inverse: Given ΔOD at two wavelengths → solve 2×2 system for Δ[HbO], Δ[HbR]
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Physical constants ───
# Extinction coefficients in 1/(cm·M)
EPS_HBO_760 = 1486.5865
EPS_HBR_760 = 3843.707
EPS_HBO_850 = 2526.391
EPS_HBR_850 = 1798.643

# Differential pathlength factors
DPF_760 = 6.0
DPF_850 = 5.5

# Source-detector distance (cm)
D = 3.0

# ─── Sampling parameters ───
FS = 10.0        # Hz
DURATION = 300.0 # seconds
N_SAMPLES = int(DURATION * FS)
T = np.arange(N_SAMPLES) / FS

# ─── Random seed for reproducibility ───
RNG = np.random.default_rng(42)


def canonical_hrf(t_hrf, peak=6.0, undershoot=16.0, ratio=6.0):
    """Generate a canonical hemodynamic response function (double-gamma)."""
    from scipy.stats import gamma as gamma_dist
    h = (gamma_dist.pdf(t_hrf, peak / 1.0, scale=1.0) -
         gamma_dist.pdf(t_hrf, undershoot / 1.0, scale=1.0) / ratio)
    h = h / np.max(np.abs(h))
    return h


def synthesize_hemodynamic_signals(t, fs):
    """Create HbO and HbR block-design time series with 3 stimulus blocks."""
    n = len(t)

    # Stimulus blocks: 3 blocks, each ~20s on, ~60s off
    stimulus = np.zeros(n)
    block_starts = [30, 110, 200]  # seconds
    block_duration = 20            # seconds
    for start in block_starts:
        i0 = int(start * fs)
        i1 = int((start + block_duration) * fs)
        stimulus[i0:min(i1, n)] = 1.0

    # HRF kernel (30 seconds long)
    t_hrf = np.arange(0, 30, 1.0 / fs)
    hrf = canonical_hrf(t_hrf)

    # Convolve stimulus with HRF
    bold = np.convolve(stimulus, hrf, mode='full')[:n]
    bold = bold / np.max(np.abs(bold))

    # HbO: positive response, peak ~5 µM
    hbo = bold * 5e-6  # in Molar

    # HbR: negative, smaller (~-1.5 µM)
    hbr = -bold * 1.5e-6

    return hbo, hbr, stimulus, block_starts, block_duration


def forward_mbll(hbo, hbr):
    """
    Forward Modified Beer-Lambert Law.
    ΔOD(λ) = ε_HbO(λ)·Δ[HbO]·DPF(λ)·d + ε_HbR(λ)·Δ[HbR]·DPF(λ)·d
    """
    od_760 = (EPS_HBO_760 * hbo * DPF_760 * D +
              EPS_HBR_760 * hbr * DPF_760 * D)
    od_850 = (EPS_HBO_850 * hbo * DPF_850 * D +
              EPS_HBR_850 * hbr * DPF_850 * D)
    return od_760, od_850


def add_noise(signal, snr_db=25):
    """Add Gaussian noise at specified SNR."""
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = RNG.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def inverse_mbll(od_760_noisy, od_850_noisy):
    """
    Inverse MBLL: solve 2×2 linear system at each time point.
    A · [HbO, HbR]^T = [ΔOD_760, ΔOD_850]^T
    """
    A = np.array([
        [EPS_HBO_760 * DPF_760 * D, EPS_HBR_760 * DPF_760 * D],
        [EPS_HBO_850 * DPF_850 * D, EPS_HBR_850 * DPF_850 * D]
    ])
    # Stack observations: shape (2, N)
    od_stack = np.stack([od_760_noisy, od_850_noisy], axis=0)
    # Solve A x = b  →  x = A^{-1} b
    A_inv = np.linalg.inv(A)
    x = A_inv @ od_stack  # (2, N)
    hbo_rec = x[0]
    hbr_rec = x[1]
    return hbo_rec, hbr_rec


def compute_psnr(gt, rec):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - rec) ** 2)
    if mse < 1e-30:
        return 100.0
    peak = np.max(np.abs(gt))
    return 10 * np.log10(peak ** 2 / mse)


def compute_cc(gt, rec):
    """Pearson correlation coefficient."""
    return float(np.corrcoef(gt, rec)[0, 1])


def compute_rmse(gt, rec):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((gt - rec) ** 2)))


def evaluate(hbo_gt, hbr_gt, hbo_rec, hbr_rec):
    """Compute metrics for both HbO and HbR."""
    metrics = {
        'HbO_PSNR_dB': compute_psnr(hbo_gt, hbo_rec),
        'HbO_CC': compute_cc(hbo_gt, hbo_rec),
        'HbO_RMSE': compute_rmse(hbo_gt, hbo_rec),
        'HbR_PSNR_dB': compute_psnr(hbr_gt, hbr_rec),
        'HbR_CC': compute_cc(hbr_gt, hbr_rec),
        'HbR_RMSE': compute_rmse(hbr_gt, hbr_rec),
    }
    # Overall averages
    metrics['PSNR_dB'] = (metrics['HbO_PSNR_dB'] + metrics['HbR_PSNR_dB']) / 2
    metrics['CC'] = (metrics['HbO_CC'] + metrics['HbR_CC']) / 2
    metrics['RMSE'] = (metrics['HbO_RMSE'] + metrics['HbR_RMSE']) / 2
    return metrics


def plot_results(t, hbo_gt, hbr_gt, od_760_noisy, od_850_noisy,
                 hbo_rec, hbr_rec, block_starts, block_duration, metrics):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Helper: shade stimulus blocks
    def shade_blocks(ax):
        for s in block_starts:
            ax.axvspan(s, s + block_duration, color='yellow', alpha=0.2, label=None)

    # Panel 1: Ground truth HbO & HbR
    ax = axes[0]
    shade_blocks(ax)
    ax.plot(t, hbo_gt * 1e6, 'r-', lw=1.5, label='HbO (GT)')
    ax.plot(t, hbr_gt * 1e6, 'b-', lw=1.5, label='HbR (GT)')
    ax.set_ylabel('Concentration (µM)')
    ax.set_title('Ground Truth Hemodynamic Response')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Noisy optical density input
    ax = axes[1]
    shade_blocks(ax)
    ax.plot(t, od_760_noisy, 'purple', lw=0.8, alpha=0.7, label='ΔOD 760nm')
    ax.plot(t, od_850_noisy, 'orange', lw=0.8, alpha=0.7, label='ΔOD 850nm')
    ax.set_ylabel('Optical Density Change')
    ax.set_title('Noisy Optical Density Input (MBLL Forward + Noise)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Recovered vs GT
    ax = axes[2]
    shade_blocks(ax)
    ax.plot(t, hbo_gt * 1e6, 'r--', lw=1.5, alpha=0.6, label='HbO (GT)')
    ax.plot(t, hbo_rec * 1e6, 'r-', lw=1.2, label='HbO (Recovered)')
    ax.plot(t, hbr_gt * 1e6, 'b--', lw=1.5, alpha=0.6, label='HbR (GT)')
    ax.plot(t, hbr_rec * 1e6, 'b-', lw=1.2, label='HbR (Recovered)')
    ax.set_ylabel('Concentration (µM)')
    ax.set_title(f'Recovered Hemodynamic Response  |  PSNR={metrics["PSNR_dB"]:.1f} dB, CC={metrics["CC"]:.4f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 4: Residuals
    ax = axes[3]
    shade_blocks(ax)
    ax.plot(t, (hbo_gt - hbo_rec) * 1e6, 'r-', lw=1.0, label='HbO residual')
    ax.plot(t, (hbr_gt - hbr_rec) * 1e6, 'b-', lw=1.0, label='HbR residual')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('Residual (µM)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Residuals (GT − Recovered)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


def main():
    print("=" * 60)
    print("Task 176: fNIRS Hemodynamic Response Recovery (MBLL)")
    print("=" * 60)

    # 1. Synthesize ground truth
    print("\n[1/5] Synthesizing hemodynamic data...")
    hbo_gt, hbr_gt, stimulus, block_starts, block_duration = synthesize_hemodynamic_signals(T, FS)
    print(f"  HbO range: [{hbo_gt.min()*1e6:.3f}, {hbo_gt.max()*1e6:.3f}] µM")
    print(f"  HbR range: [{hbr_gt.min()*1e6:.3f}, {hbr_gt.max()*1e6:.3f}] µM")

    # 2. Forward model
    print("\n[2/5] Applying forward MBLL...")
    od_760_clean, od_850_clean = forward_mbll(hbo_gt, hbr_gt)
    od_760_noisy = add_noise(od_760_clean, snr_db=25)
    od_850_noisy = add_noise(od_850_clean, snr_db=25)
    print(f"  ΔOD_760 range: [{od_760_noisy.min():.6f}, {od_760_noisy.max():.6f}]")
    print(f"  ΔOD_850 range: [{od_850_noisy.min():.6f}, {od_850_noisy.max():.6f}]")

    # 3. Inverse solve
    print("\n[3/5] Solving inverse MBLL...")
    hbo_rec, hbr_rec = inverse_mbll(od_760_noisy, od_850_noisy)
    print(f"  Recovered HbO range: [{hbo_rec.min()*1e6:.3f}, {hbo_rec.max()*1e6:.3f}] µM")
    print(f"  Recovered HbR range: [{hbr_rec.min()*1e6:.3f}, {hbr_rec.max()*1e6:.3f}] µM")

    # 4. Evaluate
    print("\n[4/5] Computing metrics...")
    metrics = evaluate(hbo_gt, hbr_gt, hbo_rec, hbr_rec)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # 5. Visualization
    print("\n[5/5] Creating visualization...")
    plot_results(T, hbo_gt, hbr_gt, od_760_noisy, od_850_noisy,
                 hbo_rec, hbr_rec, block_starts, block_duration, metrics)

    # 6. Save arrays
    gt_stack = np.stack([hbo_gt, hbr_gt], axis=0)   # (2, N)
    rec_stack = np.stack([hbo_rec, hbr_rec], axis=0) # (2, N)
    input_stack = np.stack([od_760_noisy, od_850_noisy], axis=0)

    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), gt_stack)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), rec_stack)
    np.save(os.path.join(RESULTS_DIR, 'input_data.npy'), input_stack)
    print(f"  Arrays saved to {RESULTS_DIR}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Overall PSNR: {metrics['PSNR_dB']:.2f} dB")
    print(f"  Overall CC:   {metrics['CC']:.4f}")
    print(f"  Overall RMSE: {metrics['RMSE']:.2e}")
    ok = metrics['PSNR_dB'] > 20 and metrics['CC'] > 0.9
    print(f"  PASS: {'YES' if ok else 'NO'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
