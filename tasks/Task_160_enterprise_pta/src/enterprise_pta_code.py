#!/usr/bin/env python
"""
Pulsar Timing Array (PTA) Bayesian Inference
=============================================
Simulates a pulsar timing array with 5 pulsars, each having:
  - White noise
  - Red noise (power-law spectrum)
  - Gravitational wave background (GWB, Hellings-Downs correlated)

Uses emcee MCMC to recover 3 key parameters:
  - log10_A_gw   : amplitude of GW background
  - log10_A_red  : amplitude of intrinsic red noise
  - gamma_red    : spectral index of red noise

Outputs metrics, visualizations, and numpy arrays for downstream evaluation.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import emcee

np.random.seed(42)

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Physical / simulation constants ──────────────────────────────────────────
N_PULSARS = 5
N_TOA = 200          # number of TOAs per pulsar
T_SPAN_YR = 10.0     # observation span in years
T_SPAN = T_SPAN_YR * 365.25 * 86400.0  # seconds
N_FREQ = 30          # number of Fourier frequencies for red/GW spectra

# True parameter values
TRUE_LOG10_A_GW = -14.5
TRUE_LOG10_A_RED = -13.5
TRUE_GAMMA_RED = 4.33   # typical value for spin noise
TRUE_EFAC = 1.0         # white noise multiplier
TRUE_SIGMA_WN = 1.0e-6  # white noise rms (seconds)

# MCMC settings
N_WALKERS = 16
N_STEPS = 500
N_BURN = 200

# ── Helper: Hellings-Downs correlation ───────────────────────────────────────
def hellings_downs(theta):
    """Hellings-Downs overlap reduction function."""
    x = (1.0 - np.cos(theta)) / 2.0
    if x < 1e-10:
        return 1.0
    hd = 1.5 * x * np.log(x) - 0.25 * x + 0.5
    return hd


def make_hd_matrix(positions):
    """Build the N_psr x N_psr Hellings-Downs correlation matrix."""
    n = len(positions)
    hd = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cos_theta = np.dot(positions[i], positions[j])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            val = hellings_downs(theta)
            hd[i, j] = val
            hd[j, i] = val
    return hd


# ── Helper: power-law PSD ────────────────────────────────────────────────────
def powerlaw_psd(freqs, log10_A, gamma):
    """Power-law power spectral density: S(f) = A^2/(12*pi^2) * (f/f_yr)^(-gamma) * f_yr^(-3)."""
    A = 10.0 ** log10_A
    f_yr = 1.0 / (365.25 * 86400.0)
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs / f_yr) ** (-gamma) * f_yr ** (-3)


# ── Generate Fourier design matrix ───────────────────────────────────────────
def fourier_design_matrix(toas, n_freq, T):
    """Create the Fourier design matrix F  (N_toa x 2*n_freq)."""
    N = len(toas)
    F = np.zeros((N, 2 * n_freq))
    freqs = np.arange(1, n_freq + 1) / T
    for i, f in enumerate(freqs):
        F[:, 2 * i] = np.sin(2.0 * np.pi * f * toas)
        F[:, 2 * i + 1] = np.cos(2.0 * np.pi * f * toas)
    return F, freqs


# ── Simulate PTA data ────────────────────────────────────────────────────────
def simulate_pta():
    """Simulate timing residuals for N_PULSARS pulsars."""
    # Random sky positions (unit vectors)
    positions = []
    for _ in range(N_PULSARS):
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        positions.append(np.array([sin_theta * np.cos(phi),
                                   sin_theta * np.sin(phi),
                                   cos_theta]))

    hd_matrix = make_hd_matrix(positions)
    freqs = np.arange(1, N_FREQ + 1) / T_SPAN

    # Red noise PSD per frequency
    psd_red = powerlaw_psd(freqs, TRUE_LOG10_A_RED, TRUE_GAMMA_RED)
    # GW PSD per frequency (GWB spectral index = 13/3 for circular binaries)
    psd_gw = powerlaw_psd(freqs, TRUE_LOG10_A_GW, 13.0 / 3.0)

    toas_all = []
    residuals_all = []
    F_all = []

    for p in range(N_PULSARS):
        # Uniform TOAs
        toas = np.linspace(0, T_SPAN, N_TOA)
        toas_all.append(toas)

        F, _ = fourier_design_matrix(toas, N_FREQ, T_SPAN)
        F_all.append(F)

    # Generate correlated GW Fourier coefficients across pulsars
    # For each frequency, draw correlated coefficients using HD matrix
    gw_coeffs = np.zeros((N_PULSARS, 2 * N_FREQ))
    for k in range(N_FREQ):
        amplitude = np.sqrt(psd_gw[k] * T_SPAN)
        # Cholesky of HD matrix for correlation
        L = np.linalg.cholesky(hd_matrix + 1e-10 * np.eye(N_PULSARS))
        for c in range(2):  # sin and cos
            uncorr = np.random.randn(N_PULSARS)
            corr = L @ uncorr * amplitude
            gw_coeffs[:, 2 * k + c] = corr

    for p in range(N_PULSARS):
        F = F_all[p]
        toas = toas_all[p]

        # Red noise (independent per pulsar)
        red_coeffs = np.zeros(2 * N_FREQ)
        for k in range(N_FREQ):
            amplitude = np.sqrt(psd_red[k] * T_SPAN)
            red_coeffs[2 * k] = np.random.randn() * amplitude
            red_coeffs[2 * k + 1] = np.random.randn() * amplitude

        # Total signal
        signal = F @ (red_coeffs + gw_coeffs[p])

        # White noise
        wn = np.random.randn(N_TOA) * TRUE_SIGMA_WN * TRUE_EFAC

        residuals = signal + wn
        residuals_all.append(residuals)

    return toas_all, residuals_all, F_all, freqs, hd_matrix, positions


# ── Log-likelihood ────────────────────────────────────────────────────────────
def log_likelihood(params, toas_all, residuals_all, F_all, freqs, hd_matrix):
    """Marginalised log-likelihood for PTA (Fourier-domain)."""
    log10_A_gw, log10_A_red, gamma_red = params

    psd_red = powerlaw_psd(freqs, log10_A_red, gamma_red)
    psd_gw = powerlaw_psd(freqs, log10_A_gw, 13.0 / 3.0)

    logL = 0.0
    sigma_wn_sq = (TRUE_SIGMA_WN * TRUE_EFAC) ** 2

    for p in range(N_PULSARS):
        F = F_all[p]
        r = residuals_all[p]
        N_inv = np.eye(N_TOA) / sigma_wn_sq

        # Per-pulsar phi: red + GW (diagonal part for this pulsar)
        phi_diag = np.zeros(2 * N_FREQ)
        for k in range(N_FREQ):
            phi_val = (psd_red[k] + hd_matrix[p, p] * psd_gw[k]) * T_SPAN
            phi_diag[2 * k] = phi_val
            phi_diag[2 * k + 1] = phi_val

        # Woodbury: (N + F phi F^T)^{-1} via Woodbury identity
        # Sigma = F diag(phi) F^T + N
        # log|Sigma| and r^T Sigma^{-1} r via Woodbury
        phi_inv = np.where(phi_diag > 0, 1.0 / phi_diag, 1e30)
        FNr = F.T @ (r / sigma_wn_sq)
        FNF = F.T @ F / sigma_wn_sq
        Sigma_a = FNF + np.diag(phi_inv)

        try:
            cf = cho_factor(Sigma_a)
            Sigma_a_inv_FNr = cho_solve(cf, FNr)
            log_det_Sigma_a = 2.0 * np.sum(np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            return -1e100

        # log-likelihood contribution
        rNr = np.sum(r ** 2) / sigma_wn_sq
        logL_p = -0.5 * rNr + 0.5 * FNr @ Sigma_a_inv_FNr
        logL_p -= 0.5 * log_det_Sigma_a
        logL_p -= 0.5 * np.sum(np.log(phi_diag[phi_diag > 0]))

        logL += logL_p

    return logL


# ── Prior ─────────────────────────────────────────────────────────────────────
def log_prior(params):
    """Uniform priors on parameters."""
    log10_A_gw, log10_A_red, gamma_red = params
    if not (-18.0 < log10_A_gw < -11.0):
        return -np.inf
    if not (-18.0 < log10_A_red < -11.0):
        return -np.inf
    if not (0.0 < gamma_red < 10.0):
        return -np.inf
    return 0.0


def log_posterior(params, toas_all, residuals_all, F_all, freqs, hd_matrix):
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, toas_all, residuals_all, F_all, freqs, hd_matrix)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PTA Bayesian Inference — Enterprise-style")
    print("=" * 60)

    # 1. Simulate data
    print("\n[1/4] Simulating PTA data ...")
    toas_all, residuals_all, F_all, freqs, hd_matrix, positions = simulate_pta()
    print(f"  → {N_PULSARS} pulsars, {N_TOA} TOAs each, {T_SPAN_YR} yr span")

    # 2. Run MCMC
    print(f"\n[2/4] Running MCMC ({N_WALKERS} walkers × {N_STEPS} steps) ...")
    true_params = np.array([TRUE_LOG10_A_GW, TRUE_LOG10_A_RED, TRUE_GAMMA_RED])
    ndim = 3

    # Initialise walkers near truth (with scatter)
    p0 = true_params + 0.3 * np.random.randn(N_WALKERS, ndim)
    # Clip to prior bounds
    p0[:, 0] = np.clip(p0[:, 0], -17.9, -11.1)
    p0[:, 1] = np.clip(p0[:, 1], -17.9, -11.1)
    p0[:, 2] = np.clip(p0[:, 2], 0.1, 9.9)

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, log_posterior,
        args=(toas_all, residuals_all, F_all, freqs, hd_matrix)
    )
    sampler.run_mcmc(p0, N_STEPS, progress=True)
    print("  → MCMC complete")

    # 3. Analyse chains
    print("\n[3/4] Analysing posterior samples ...")
    samples = sampler.get_chain(discard=N_BURN, flat=True)
    param_names = ["log10_A_gw", "log10_A_red", "gamma_red"]
    medians = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)

    print(f"\n  {'Parameter':<15s} {'True':>10s} {'Median':>10s} {'Std':>10s}")
    print("  " + "-" * 50)
    for i, name in enumerate(param_names):
        print(f"  {name:<15s} {true_params[i]:10.3f} {medians[i]:10.3f} {stds[i]:10.3f}")

    # Relative errors
    re_values = {}
    for i, name in enumerate(param_names):
        if abs(true_params[i]) > 1e-10:
            re = abs(medians[i] - true_params[i]) / abs(true_params[i])
        else:
            re = abs(medians[i] - true_params[i])
        re_values[name] = float(re)

    # Compute power spectrum comparison (red noise PSD)
    psd_true = powerlaw_psd(freqs, TRUE_LOG10_A_RED, TRUE_GAMMA_RED)
    psd_recon = powerlaw_psd(freqs, medians[1], medians[2])
    # Cross-correlation of log-PSDs
    log_psd_true = np.log10(psd_true + 1e-100)
    log_psd_recon = np.log10(psd_recon + 1e-100)
    cc_num = np.sum((log_psd_true - log_psd_true.mean()) *
                     (log_psd_recon - log_psd_recon.mean()))
    cc_den = np.sqrt(np.sum((log_psd_true - log_psd_true.mean()) ** 2) *
                     np.sum((log_psd_recon - log_psd_recon.mean()) ** 2))
    psd_cc = float(cc_num / (cc_den + 1e-30))

    # GW background PSD comparison
    psd_gw_true = powerlaw_psd(freqs, TRUE_LOG10_A_GW, 13.0 / 3.0)
    psd_gw_recon = powerlaw_psd(freqs, medians[0], 13.0 / 3.0)
    log_gw_true = np.log10(psd_gw_true + 1e-100)
    log_gw_recon = np.log10(psd_gw_recon + 1e-100)
    cc_gw_num = np.sum((log_gw_true - log_gw_true.mean()) *
                        (log_gw_recon - log_gw_recon.mean()))
    cc_gw_den = np.sqrt(np.sum((log_gw_true - log_gw_true.mean()) ** 2) *
                        np.sum((log_gw_recon - log_gw_recon.mean()) ** 2))
    psd_gw_cc = float(cc_gw_num / (cc_gw_den + 1e-30))

    mean_re = float(np.mean(list(re_values.values())))

    print(f"\n  Mean relative error: {mean_re:.4f}")
    print(f"  Red noise PSD CC:   {psd_cc:.4f}")
    print(f"  GW PSD CC:          {psd_gw_cc:.4f}")

    # 4. Save outputs
    print("\n[4/4] Saving results ...")

    # Metrics
    metrics = {
        "log10_A_gw_true": TRUE_LOG10_A_GW,
        "log10_A_gw_recovered": float(medians[0]),
        "log10_A_gw_std": float(stds[0]),
        "log10_A_gw_RE": re_values["log10_A_gw"],
        "log10_A_red_true": TRUE_LOG10_A_RED,
        "log10_A_red_recovered": float(medians[1]),
        "log10_A_red_std": float(stds[1]),
        "log10_A_red_RE": re_values["log10_A_red"],
        "gamma_red_true": TRUE_GAMMA_RED,
        "gamma_red_recovered": float(medians[2]),
        "gamma_red_std": float(stds[2]),
        "gamma_red_RE": re_values["gamma_red"],
        "mean_parameter_RE": mean_re,
        "red_noise_PSD_CC": psd_cc,
        "GW_PSD_CC": psd_gw_cc,
        "n_pulsars": N_PULSARS,
        "n_toa": N_TOA,
        "n_walkers": N_WALKERS,
        "n_steps": N_STEPS,
        "n_burn": N_BURN,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  → metrics.json saved")

    # Ground truth & reconstruction arrays
    gt_dict = {
        "true_params": true_params,
        "freqs": freqs,
        "psd_red_true": psd_true,
        "psd_gw_true": psd_gw_true,
        "residuals": [r for r in residuals_all],
        "hd_matrix": hd_matrix,
    }
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_dict,
            allow_pickle=True)

    recon_dict = {
        "recovered_params": medians,
        "param_stds": stds,
        "psd_red_recon": psd_recon,
        "psd_gw_recon": psd_gw_recon,
        "samples": samples,
    }
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_dict,
            allow_pickle=True)
    print("  → ground_truth.npy, reconstruction.npy saved")

    # ── Visualization ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Trace plots
    chain = sampler.get_chain()
    for i, name in enumerate(param_names):
        ax = axes[0, 0] if i == 0 else (axes[0, 1] if i == 1 else axes[0, 2])
        for w in range(N_WALKERS):
            ax.plot(chain[:, w, i], alpha=0.3, lw=0.5)
        ax.axhline(true_params[i], color='r', lw=2, label='Truth')
        ax.axhline(medians[i], color='blue', ls='--', lw=1.5, label='Median')
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(f'Trace: {name}')
        ax.legend(fontsize=8)

    # (1,0) Red noise PSD
    ax = axes[1, 0]
    ax.loglog(freqs * 365.25 * 86400, psd_true, 'r-', lw=2, label='True red noise')
    ax.loglog(freqs * 365.25 * 86400, psd_recon, 'b--', lw=2, label='Recovered')
    ax.set_xlabel('Frequency (1/yr)')
    ax.set_ylabel('PSD (s²/Hz)')
    ax.set_title(f'Red Noise PSD (CC={psd_cc:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) GW PSD
    ax = axes[1, 1]
    ax.loglog(freqs * 365.25 * 86400, psd_gw_true, 'r-', lw=2, label='True GWB')
    ax.loglog(freqs * 365.25 * 86400, psd_gw_recon, 'b--', lw=2, label='Recovered')
    ax.set_xlabel('Frequency (1/yr)')
    ax.set_ylabel('PSD (s²/Hz)')
    ax.set_title(f'GW Background PSD (CC={psd_gw_cc:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Corner-like: 2D posterior (A_gw vs A_red)
    ax = axes[1, 2]
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.2, c='steelblue')
    ax.axvline(TRUE_LOG10_A_GW, color='r', lw=1.5, label='True')
    ax.axhline(TRUE_LOG10_A_RED, color='r', lw=1.5)
    ax.scatter([medians[0]], [medians[1]], c='blue', marker='x', s=100,
               zorder=5, label='Median')
    ax.set_xlabel('log10_A_gw')
    ax.set_ylabel('log10_A_red')
    ax.set_title('Posterior: A_gw vs A_red')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {fig_path} saved")

    print("\n" + "=" * 60)
    print("DONE — PTA Bayesian inference complete")
    print(f"  Mean param RE = {mean_re:.4f}")
    print(f"  Red PSD CC    = {psd_cc:.4f}")
    print(f"  GW PSD CC     = {psd_gw_cc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
