"""
Task 181: Seismic dv/v Estimation via Stretching Method

Inverse Problem: Inverting seismic velocity change (dv/v) from ambient noise
cross-correlation using the stretching technique.

Forward: CCF_cur(t) = interp(CCF_ref, t*(1+eps)) where eps = -dv/v
Inverse: For trial eps values, stretch reference, compute CC with current CCF,
         find eps that maximizes CC => dv/v_est = -eps_best
"""

import json
import os
import numpy as np
from scipy.interpolate import interp1d

# ─────────────────────────── Parameters ───────────────────────────
SEED = 42
FS = 100.0            # sampling rate (Hz)
T_START, T_END = -10.0, 10.0
F0 = 2.0              # Ricker wavelet center frequency
N_DAYS = 50           # number of synthetic observation days
DVV_AMP = 0.002       # 0.2% amplitude of dv/v sinusoidal variation
DVV_NOISE = 0.0005    # 0.05% noise level
DVV_PERIOD = 30.0     # period of sinusoidal dv/v in days
NOISE_LEVEL = 0.02    # measurement noise added to perturbed CCFs
STRETCH_RANGE = 0.01  # ±1% trial stretching
STRETCH_STEPS = 201   # number of trial epsilon values
CODA_TMIN = 1.0       # coda window: |t| > CODA_TMIN seconds

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


# ─────────────────────── Synthesise Reference CCF ──────────────────
def make_ricker(t: np.ndarray, f0: float) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet centred at t=0."""
    a = 1.0 / f0
    u = (np.pi * (t) / a) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


def synthesise_reference_ccf(t: np.ndarray, f0: float,
                              decay_tau: float = 5.0,
                              n_scatterers: int = 12) -> np.ndarray:
    """
    Build a realistic-looking reference cross-correlation function.
    Ricker wavelet modulated by exponential decay + scattered arrivals.
    """
    ccf = make_ricker(t, f0) * np.exp(-np.abs(t) / decay_tau)
    # Add scattered arrivals (smaller Ricker wavelets at random times)
    for _ in range(n_scatterers):
        t_shift = rng.uniform(-8, 8)
        amp = rng.uniform(0.05, 0.25) * np.exp(-np.abs(t_shift) / decay_tau)
        f_scat = rng.uniform(1.5, 3.5)
        ccf += amp * make_ricker(t - t_shift, f_scat)
    return ccf


# ─────────────────────── Forward Operator ──────────────────────────
def forward_stretch(ccf_ref: np.ndarray, t: np.ndarray,
                    dvv: float) -> np.ndarray:
    """
    Forward operator: apply velocity change to reference CCF.
    CCF_cur(t) = CCF_ref(t * (1 + eps))  where eps = -dv/v
    """
    eps = -dvv
    t_stretched = t * (1.0 + eps)
    interp_func = interp1d(t, ccf_ref, kind='cubic',
                           bounds_error=False, fill_value=0.0)
    return interp_func(t_stretched)


# ─────────────────── Inverse Solver (Stretching) ───────────────────
def inverse_stretching(ccf_ref: np.ndarray, ccf_cur: np.ndarray,
                       t: np.ndarray, stretch_range: float,
                       stretch_steps: int,
                       coda_tmin: float) -> tuple:
    """
    Stretching method: find epsilon that maximises correlation between
    stretched reference and current CCF within the coda window.

    Returns (dv/v_est, cc_best, cc_curve)
    """
    # Coda window mask: |t| > coda_tmin
    coda_mask = np.abs(t) >= coda_tmin

    trial_eps = np.linspace(-stretch_range, stretch_range, stretch_steps)
    interp_func = interp1d(t, ccf_ref, kind='cubic',
                           bounds_error=False, fill_value=0.0)

    cc_values = np.empty(stretch_steps)
    cur_coda = ccf_cur[coda_mask]
    cur_coda_demean = cur_coda - cur_coda.mean()
    cur_norm = np.sqrt(np.sum(cur_coda_demean ** 2))

    for i, eps in enumerate(trial_eps):
        t_stretched = t * (1.0 + eps)
        ref_stretched = interp_func(t_stretched)
        ref_coda = ref_stretched[coda_mask]
        ref_coda_demean = ref_coda - ref_coda.mean()
        ref_norm = np.sqrt(np.sum(ref_coda_demean ** 2))
        if ref_norm < 1e-15 or cur_norm < 1e-15:
            cc_values[i] = 0.0
        else:
            cc_values[i] = np.sum(ref_coda_demean * cur_coda_demean) / (
                ref_norm * cur_norm)

    best_idx = np.argmax(cc_values)
    eps_best = trial_eps[best_idx]
    dvv_est = -eps_best
    return dvv_est, cc_values[best_idx], cc_values


# ──────────────────── Synthesise Multi-day Dataset ─────────────────
def synthesise_dataset(t, ccf_ref, n_days, dvv_amp, dvv_noise_std,
                       dvv_period, noise_level):
    """Create synthetic dv/v time series and perturbed CCFs."""
    days = np.arange(n_days)
    dvv_true = dvv_amp * np.sin(2.0 * np.pi * days / dvv_period) + \
        dvv_noise_std * rng.standard_normal(n_days)

    ccf_matrix = np.empty((n_days, len(t)))
    for d in range(n_days):
        ccf_cur = forward_stretch(ccf_ref, t, dvv_true[d])
        ccf_cur += noise_level * rng.standard_normal(len(t)) * np.max(
            np.abs(ccf_ref))
        ccf_matrix[d] = ccf_cur

    return days, dvv_true, ccf_matrix


# ──────────────────────── Evaluation Metrics ───────────────────────
def compute_metrics(dvv_true: np.ndarray,
                    dvv_est: np.ndarray) -> dict:
    """Compute dv/v estimation quality metrics."""
    # Mean absolute error
    mae = float(np.mean(np.abs(dvv_est - dvv_true)))

    # Relative error (fraction of amplitude range)
    amp_range = np.max(dvv_true) - np.min(dvv_true)
    rel_error = float(mae / amp_range) if amp_range > 0 else float('inf')

    # Correlation coefficient
    cc = float(np.corrcoef(dvv_true, dvv_est)[0, 1])

    # PSNR (treating dv/v time series as 1D signal)
    mse = float(np.mean((dvv_est - dvv_true) ** 2))
    peak = float(np.max(np.abs(dvv_true)))
    if mse > 0 and peak > 0:
        psnr = float(20.0 * np.log10(peak / np.sqrt(mse)))
    else:
        psnr = float('inf')

    return {
        "dvv_mae": mae,
        "dvv_relative_error": rel_error,
        "dvv_correlation_coefficient": cc,
        "dvv_psnr_dB": psnr,
    }


# ──────────────────────── Visualisation ────────────────────────────
def plot_results(t, ccf_ref, ccf_cur_example, days, dvv_true, dvv_est,
                 save_path):
    """Multi-panel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Reference CCF
    ax = axes[0, 0]
    ax.plot(t, ccf_ref, 'k', lw=0.8)
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("(a) Reference CCF")
    ax.set_xlim(t[0], t[-1])

    # (b) Reference vs current (one example day)
    ax = axes[0, 1]
    ax.plot(t, ccf_ref, 'k', lw=0.8, label="Reference")
    ax.plot(t, ccf_cur_example, 'r', lw=0.8, alpha=0.7, label="Current (day 10)")
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("(b) Reference vs Perturbed CCF")
    ax.legend(fontsize=9)
    ax.set_xlim(t[0], t[-1])

    # (c) True vs estimated dv/v
    ax = axes[1, 0]
    ax.plot(days, dvv_true * 100, 'k-o', ms=3, lw=1.2, label="True dv/v")
    ax.plot(days, dvv_est * 100, 'r-s', ms=3, lw=1.2, label="Estimated dv/v")
    ax.set_xlabel("Day")
    ax.set_ylabel("dv/v (%)")
    ax.set_title("(c) dv/v Time Series")
    ax.legend(fontsize=9)

    # (d) Residual
    ax = axes[1, 1]
    residual = (dvv_true - dvv_est) * 100
    ax.bar(days, residual, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("Residual dv/v (%)")
    ax.set_title("(d) Residual (True − Estimated)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {save_path}")


# ──────────────────────── Main Pipeline ────────────────────────────
def main():
    # Time axis
    n_samples = int((T_END - T_START) * FS) + 1
    t = np.linspace(T_START, T_END, n_samples)

    # Step 1: Synthesise reference CCF
    ccf_ref = synthesise_reference_ccf(t, F0)
    print(f"Reference CCF: {len(ccf_ref)} samples, "
          f"t=[{t[0]:.1f}, {t[-1]:.1f}] s")

    # Step 2: Generate multi-day dataset with known dv/v
    days, dvv_true, ccf_matrix = synthesise_dataset(
        t, ccf_ref, N_DAYS, DVV_AMP, DVV_NOISE, DVV_PERIOD, NOISE_LEVEL)
    print(f"Synthesised {N_DAYS} days of CCF data")
    print(f"True dv/v range: [{dvv_true.min()*100:.4f}%, "
          f"{dvv_true.max()*100:.4f}%]")

    # Step 3: Invert dv/v using stretching method
    dvv_est = np.empty(N_DAYS)
    cc_best_arr = np.empty(N_DAYS)
    for d in range(N_DAYS):
        dvv_est[d], cc_best_arr[d], _ = inverse_stretching(
            ccf_ref, ccf_matrix[d], t,
            STRETCH_RANGE, STRETCH_STEPS, CODA_TMIN)

    print(f"Estimated dv/v range: [{dvv_est.min()*100:.4f}%, "
          f"{dvv_est.max()*100:.4f}%]")
    print(f"Mean best CC: {cc_best_arr.mean():.4f}")

    # Step 4: Evaluate
    metrics = compute_metrics(dvv_true, dvv_est)
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Step 5: Visualise
    plot_results(t, ccf_ref, ccf_matrix[10], days, dvv_true, dvv_est,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    # Step 6: Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), dvv_true)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), dvv_est)
    print("Arrays saved.")


if __name__ == "__main__":
    main()
