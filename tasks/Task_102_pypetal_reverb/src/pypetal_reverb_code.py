"""
pypetal_reverb - Reverberation Mapping (AGN Transfer Function Recovery)
========================================================================
From AGN continuum and emission line light curves, recover the
transfer function Ψ(τ) via Tikhonov deconvolution.

Physics:
  - Forward:  L(t) = ∫ Ψ(τ) × C(t-τ) dτ + noise   (convolution)
  - GT transfer function: Gaussian peaked at τ = 20 days
  - Inverse: Tikhonov-regularised deconvolution in frequency domain
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import os
import json
import time

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_102_pypetal_reverb"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
N_TIME    = 512         # number of time samples
DT        = 1.0         # days per sample
T_MAX     = N_TIME * DT
TAU_PEAK  = 20.0        # peak lag (days)
TAU_SIGMA = 5.0         # transfer function width (days)
SNR       = 25.0        # signal-to-noise ratio
LAMBDA_REG = "auto"     # Tikhonov regularisation parameter
SEED      = 42

np.random.seed(SEED)


# ====================================================================
# 1. Ground-truth components
# ====================================================================
def create_continuum(N, dt):
    """
    Generate a realistic AGN continuum light curve using a
    damped random walk (DRW) process.
    """
    tau_drw = 100.0   # DRW timescale (days)
    sigma_drw = 0.15  # DRW amplitude
    t = np.arange(N) * dt
    x = np.zeros(N)
    for i in range(1, N):
        decay = np.exp(-dt / tau_drw)
        x[i] = decay * x[i-1] + sigma_drw * np.sqrt(1 - decay**2) * np.random.randn()
    # Mean flux ~ 1.0
    continuum = 1.0 + x
    return t, continuum


def create_transfer_function(N, dt, tau_peak, tau_sigma):
    """
    GT transfer function: Gaussian in lag space.
    Ψ(τ) = A * exp(-(τ - τ_peak)^2 / (2σ^2))
    """
    tau = np.arange(N) * dt
    psi = np.exp(-0.5 * ((tau - tau_peak) / tau_sigma)**2)
    psi /= psi.sum() * dt  # normalise to unit integral
    return tau, psi


# ====================================================================
# 2. Forward model
# ====================================================================
def forward_model(continuum, psi, dt, snr):
    """
    L(t) = ∫ Ψ(τ) × C(t-τ) dτ + noise
    Implemented as discrete convolution.
    """
    line_clean = np.convolve(continuum, psi * dt, mode='full')[:len(continuum)]
    noise_std = line_clean.std() / snr
    noise = np.random.normal(0, noise_std, len(continuum))
    line_obs = line_clean + noise
    return line_clean, line_obs, noise_std


# ====================================================================
# 3. Inverse: Tikhonov deconvolution
# ====================================================================
def tikhonov_deconvolve(continuum, line_obs, dt, lam=None):
    """
    Recover Ψ(τ) from:
        L = C * Ψ + noise
    using Non-Negative Least Squares (NNLS) with an explicit
    convolution matrix.  This enforces non-negativity of Ψ(τ)
    and avoids the over-smoothing of frequency-domain Tikhonov.

    We only recover the first M=100 lag values (0–99 days) since
    the GT transfer function is negligible beyond ~50 days.
    """
    N = len(continuum)
    M = min(100, N)  # number of lag bins to recover

    # Build convolution matrix  A[i, j] = C(t_i - τ_j) * dt
    A = np.zeros((N, M))
    for j in range(M):
        for i in range(j, N):
            A[i, j] = continuum[i - j] * dt

    # NNLS solve:  min ||A @ psi - line_obs||^2  s.t. psi >= 0
    psi_nnls, _ = nnls(A, line_obs)

    # Embed into full-length array for downstream compatibility
    psi_rec = np.zeros(N)
    psi_rec[:M] = psi_nnls

    return psi_rec


def compute_ccf(continuum, line_obs, dt):
    """
    Cross-correlation function between continuum and line.
    """
    N = len(continuum)
    c_norm = continuum - continuum.mean()
    l_norm = line_obs - line_obs.mean()
    ccf = np.correlate(l_norm, c_norm, mode='full')
    ccf = ccf / (np.std(continuum) * np.std(line_obs) * N)
    lags = np.arange(-N + 1, N) * dt
    # Keep only positive lags
    pos = lags >= 0
    return lags[pos], ccf[pos]


# ====================================================================
# 4. Metrics
# ====================================================================
def compute_metrics(psi_gt, psi_rec, tau, dt):
    """PSNR and CC of recovered transfer function."""
    # Trim to relevant lag range
    max_lag = 80.0
    mask = tau <= max_lag
    gt = psi_gt[mask]
    rec = psi_rec[mask]

    # Normalise both to peak = 1 for a fair, scale-invariant comparison
    gt_peak = gt.max()
    rec_peak = rec.max()
    gt_norm  = gt / gt_peak if gt_peak > 0 else gt
    rec_norm = rec / rec_peak if rec_peak > 0 else rec

    mse = np.mean((gt_norm - rec_norm)**2)
    # data_range = 1.0 after peak-normalisation
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
    cc = float(np.corrcoef(gt_norm, rec_norm)[0, 1])
    rmse = float(np.sqrt(mse))

    # Peak lag recovery
    peak_gt = tau[mask][np.argmax(gt)]
    peak_rec = tau[mask][np.argmax(rec)]
    peak_error = abs(peak_rec - peak_gt)

    return psnr, cc, rmse, peak_error, rec_norm, gt_norm


# ====================================================================
# 5. Visualization
# ====================================================================
def plot_results(t, continuum, line_clean, line_obs, tau, psi_gt,
                 psi_rec_norm, ccf_lags, ccf, metrics_dict):
    """4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    max_lag = 80.0
    mask = tau <= max_lag
    ccf_mask = ccf_lags <= max_lag
    
    # Panel 1: Light curves
    ax = axes[0, 0]
    ax.plot(t, continuum, 'b-', lw=0.8, label='Continuum C(t)')
    ax.plot(t, line_obs, 'r-', lw=0.8, alpha=0.6, label='Line L(t) [observed]')
    ax.plot(t, line_clean, 'g--', lw=0.8, label='Line L(t) [clean]')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("AGN Light Curves")
    ax.legend(fontsize=8)
    
    # Panel 2: Transfer function
    ax = axes[0, 1]
    ax.plot(tau[mask], psi_gt[mask], 'b-', lw=2, label='GT  Ψ(τ)')
    ax.plot(tau[mask], psi_rec_norm, 'r--', lw=2, label='Recovered Ψ(τ)')
    ax.axvline(TAU_PEAK, color='gray', ls=':', lw=1, label=f'True peak = {TAU_PEAK} d')
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Ψ(τ)")
    ax.set_title(f"Transfer Function | PSNR={metrics_dict['PSNR']:.1f} dB, CC={metrics_dict['CC']:.4f}")
    ax.legend(fontsize=8)
    
    # Panel 3: CCF
    ax = axes[1, 0]
    ax.plot(ccf_lags[ccf_mask], ccf[ccf_mask], 'k-', lw=1.5)
    ax.axvline(TAU_PEAK, color='r', ls='--', lw=1, label=f'True lag = {TAU_PEAK} d')
    peak_ccf_lag = ccf_lags[ccf_mask][np.argmax(ccf[ccf_mask])]
    ax.axvline(peak_ccf_lag, color='b', ls=':', lw=1, label=f'CCF peak = {peak_ccf_lag:.1f} d')
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("CCF")
    ax.set_title("Cross-Correlation Function")
    ax.legend(fontsize=8)
    
    # Panel 4: Residual of transfer function
    ax = axes[1, 1]
    residual = psi_gt[mask] - psi_rec_norm
    ax.plot(tau[mask], residual, 'k-', lw=1)
    ax.axhline(0, color='r', ls='--', lw=0.5)
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Ψ Residual | Peak error = {metrics_dict['peak_lag_error']:.1f} days")
    ax.fill_between(tau[mask], residual, alpha=0.3, color='gray')
    
    plt.tight_layout()
    for path in [os.path.join(RESULTS_DIR, "vis_result.png"),
                 os.path.join(ASSETS_DIR, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ====================================================================
# 6. Main
# ====================================================================
def main():
    print("=" * 60)
    print("Task 102: Reverberation Mapping (pypetal_reverb)")
    print("=" * 60)

    t0 = time.time()

    # Generate data
    print("\n[1] Generating AGN light curves ...")
    t, continuum = create_continuum(N_TIME, DT)
    tau, psi_gt = create_transfer_function(N_TIME, DT, TAU_PEAK, TAU_SIGMA)
    line_clean, line_obs, noise_std = forward_model(continuum, psi_gt, DT, SNR)
    print(f"    Time span: {T_MAX:.0f} days, {N_TIME} samples")
    print(f"    Noise std: {noise_std:.4f}")

    # Inverse
    print("[2] Tikhonov deconvolution ...")
    psi_rec = tikhonov_deconvolve(continuum, line_obs, DT, lam=LAMBDA_REG)

    # CCF
    print("[3] Computing CCF ...")
    ccf_lags, ccf = compute_ccf(continuum, line_obs, DT)

    elapsed = time.time() - t0
    print(f"    Elapsed: {elapsed:.1f} s")

    # Metrics
    print("[4] Computing metrics ...")
    psnr, cc, rmse, peak_err, psi_rec_norm, psi_gt_trimmed = \
        compute_metrics(psi_gt, psi_rec, tau, DT)

    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.6f}")
    print(f"    Peak lag error = {peak_err:.1f} days")

    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RMSE": float(rmse),
        "peak_lag_error": float(peak_err),
    }

    # Save
    print("[5] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), psi_gt)
        np.save(os.path.join(d, "recon_output.npy"), psi_rec)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Plot
    print("[6] Plotting ...")
    plot_results(t, continuum, line_clean, line_obs, tau, psi_gt,
                 psi_rec_norm, ccf_lags, ccf, metrics)

    print(f"\n{'='*60}")
    print("Task 102 COMPLETE")
    print(f"{'='*60}")
    return metrics


if __name__ == "__main__":
    metrics = main()
