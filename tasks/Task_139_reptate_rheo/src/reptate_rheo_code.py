"""
reptate_rheo — Polymer Rheology Inversion Pipeline
====================================================
Task 139: Invert dynamic mechanical data (G'(ω), G''(ω)) to recover
polymer molecular weight distribution / chain topology parameters.

Repo reference: https://github.com/jorge-ramirez-upm/RepTate

Physics:
    Rouse model for unentangled polymer dynamics:
        G'(ω)  = G0 * Σ_{p=1}^{N} ω²τ_p² / (1 + ω²τ_p²)
        G''(ω) = G0 * Σ_{p=1}^{N} ωτ_p   / (1 + ω²τ_p²)  + ω η_s
    where τ_p = τ_R / p² are the Rouse relaxation times.

Inverse problem:
    Given noisy G'(ω) and G''(ω), recover (G0, τ_R, η_s).

Method:
    Global optimisation (differential evolution) in log-parameter space
    with log-space residuals to handle data spanning multiple decades.

Usage:
    /data/yjh/reptate_rheo_env/bin/python reptate_rheo_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import differential_evolution

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
# Forward Model: Rouse model for polymer viscoelasticity
# ════════════════════════════════════════════════════════════

def rouse_model(omega, G0, tau_R, N_modes, eta_s=0.0):
    """
    Compute storage (G') and loss (G'') moduli using the Rouse model.

    Parameters
    ----------
    omega : ndarray
        Angular frequencies (rad/s).
    G0 : float
        Modulus prefactor nkT (Pa).
    tau_R : float
        Longest Rouse relaxation time (s).
    N_modes : int
        Number of Rouse modes to sum.
    eta_s : float
        Solvent viscosity contribution (Pa·s).

    Returns
    -------
    G_prime, G_double_prime : ndarray
        Storage and loss moduli.
    """
    omega = np.asarray(omega, dtype=np.float64)
    G_prime = np.zeros_like(omega)
    G_double_prime = np.zeros_like(omega)

    for p in range(1, N_modes + 1):
        tau_p = tau_R / p**2
        wt = omega * tau_p
        wt2 = wt * wt
        denom = 1.0 + wt2
        G_prime += G0 * wt2 / denom
        G_double_prime += G0 * wt / denom

    G_double_prime += omega * eta_s
    return G_prime, G_double_prime


# ════════════════════════════════════════════════════════════
# Data Generation
# ════════════════════════════════════════════════════════════

def generate_data(noise_level=0.05, seed=42):
    """
    Generate synthetic rheology data with known ground-truth parameters.

    Returns
    -------
    omega, G_prime_obs, G_double_prime_obs,
    G_prime_true, G_double_prime_true, true_params
    """
    true_params = {
        'G0': 1.5e5,       # Pa  – modulus prefactor
        'tau_R': 0.01,      # s   – longest Rouse time
        'N_modes': 20,      # –   – number of Rouse modes
        'eta_s': 50.0,      # Pa·s – solvent viscosity
    }

    # Frequency sweep: 5 decades, 100 points (log-spaced)
    omega = np.logspace(-2, 4, 100)

    G_prime_true, G_double_prime_true = rouse_model(
        omega,
        true_params['G0'],
        true_params['tau_R'],
        true_params['N_modes'],
        true_params['eta_s'],
    )

    # Multiplicative Gaussian noise
    rng = np.random.default_rng(seed)
    G_prime_obs = G_prime_true * (1.0 + noise_level * rng.standard_normal(len(omega)))
    G_double_prime_obs = G_double_prime_true * (1.0 + noise_level * rng.standard_normal(len(omega)))

    # Enforce positivity
    G_prime_obs = np.maximum(G_prime_obs, 1e-3)
    G_double_prime_obs = np.maximum(G_double_prime_obs, 1e-3)

    return omega, G_prime_obs, G_double_prime_obs, G_prime_true, G_double_prime_true, true_params


# ════════════════════════════════════════════════════════════
# Inverse Problem
# ════════════════════════════════════════════════════════════

def _objective(params_vec, omega, G_prime_obs, G_double_prime_obs, N_modes):
    """Log-space least-squares objective (handles multi-decade data well)."""
    log_G0, log_tau_R, log_eta_s = params_vec
    G0 = 10.0 ** log_G0
    tau_R = 10.0 ** log_tau_R
    eta_s = 10.0 ** log_eta_s

    G_p, G_pp = rouse_model(omega, G0, tau_R, N_modes, eta_s)

    EPS = 1e-30
    res_p = np.log10(G_p + EPS) - np.log10(G_prime_obs + EPS)
    res_pp = np.log10(G_pp + EPS) - np.log10(G_double_prime_obs + EPS)

    return float(np.sum(res_p**2 + res_pp**2))


def reconstruct(omega, G_prime_obs, G_double_prime_obs, N_modes=20):
    """
    Recover Rouse model parameters from noisy G'/G'' data.

    Uses differential evolution (global) followed by L-BFGS-B polish (local).

    Returns
    -------
    G_prime_fit, G_double_prime_fit, fitted_params
    """
    bounds = [
        (3.0, 7.0),    # log10(G0)   : 1e3 → 1e7  Pa
        (-4.0, 1.0),   # log10(τ_R)  : 1e-4 → 10  s
        (-1.0, 4.0),   # log10(η_s)  : 0.1 → 1e4  Pa·s
    ]

    result = differential_evolution(
        _objective,
        bounds,
        args=(omega, G_prime_obs, G_double_prime_obs, N_modes),
        seed=42,
        maxiter=2000,
        tol=1e-12,
        polish=True,
        popsize=25,
    )

    G0_fit = 10.0 ** result.x[0]
    tau_R_fit = 10.0 ** result.x[1]
    eta_s_fit = 10.0 ** result.x[2]

    fitted_params = {
        'G0': float(G0_fit),
        'tau_R': float(tau_R_fit),
        'N_modes': int(N_modes),
        'eta_s': float(eta_s_fit),
    }

    G_prime_fit, G_double_prime_fit = rouse_model(
        omega, G0_fit, tau_R_fit, N_modes, eta_s_fit,
    )

    return G_prime_fit, G_double_prime_fit, fitted_params


# ════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════

def compute_metrics(true_params, fitted_params,
                    G_true_p, G_true_pp, G_fit_p, G_fit_pp):
    """Compute parameter errors, spectral PSNR, and correlation coefficient."""
    # Per-parameter relative errors
    param_errors = {}
    for key in ('G0', 'tau_R', 'eta_s'):
        tv = true_params[key]
        fv = fitted_params[key]
        re = abs(tv - fv) / abs(tv)
        param_errors[key] = {'true': float(tv), 'fitted': float(fv), 'rel_error': float(re)}

    mean_re = float(np.mean([v['rel_error'] for v in param_errors.values()]))

    # Concatenate G' and G'' for spectral metrics (log scale)
    EPS = 1e-30
    log_true = np.log10(np.concatenate([G_true_p, G_true_pp]) + EPS)
    log_fit = np.log10(np.concatenate([G_fit_p, G_fit_pp]) + EPS)

    data_range = float(log_true.max() - log_true.min())
    mse = float(np.mean((log_true - log_fit) ** 2))
    psnr = 10.0 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')

    cc = float(np.corrcoef(log_true, log_fit)[0, 1])

    return {
        'psnr_dB': float(psnr),
        'correlation_coefficient': cc,
        'mean_parameter_relative_error': mean_re,
        'parameters': param_errors,
        'method': 'Rouse_model_differential_evolution_fitting',
    }


# ════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════

def visualize(omega, G_p_obs, G_pp_obs, G_p_true, G_pp_true,
              G_p_fit, G_pp_fit, metrics, save_path):
    """Create a 2×2 diagnostic plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Storage modulus G'
    ax = axes[0, 0]
    ax.loglog(omega, G_p_true, 'b-', lw=2, label="G' (true)")
    ax.loglog(omega, G_p_obs, 'rx', ms=4, alpha=0.5, label="G' (observed)")
    ax.loglog(omega, G_p_fit, 'g--', lw=2, label="G' (fitted)")
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel("G' (Pa)", fontsize=11)
    ax.set_title("(a) Storage Modulus G'", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # (b) Loss modulus G''
    ax = axes[0, 1]
    ax.loglog(omega, G_pp_true, 'b-', lw=2, label="G'' (true)")
    ax.loglog(omega, G_pp_obs, 'rx', ms=4, alpha=0.5, label="G'' (observed)")
    ax.loglog(omega, G_pp_fit, 'g--', lw=2, label="G'' (fitted)")
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel("G'' (Pa)", fontsize=11)
    ax.set_title("(b) Loss Modulus G''", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # (c) Parameter comparison (bar chart, log scale)
    ax = axes[1, 0]
    params = metrics['parameters']
    names = list(params.keys())
    true_vals = [params[n]['true'] for n in names]
    fit_vals = [params[n]['fitted'] for n in names]
    x_pos = np.arange(len(names))
    w = 0.35
    ax.bar(x_pos - w / 2, true_vals, w, label='True', color='steelblue', alpha=0.8)
    ax.bar(x_pos + w / 2, fit_vals, w, label='Fitted', color='seagreen', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yscale('log')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(c) Parameter Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # (d) Residuals (% relative error vs true, noise-free)
    ax = axes[1, 1]
    res_p = (G_p_fit - G_p_true) / G_p_true * 100.0
    res_pp = (G_pp_fit - G_pp_true) / G_pp_true * 100.0
    ax.semilogx(omega, res_p, 'b.-', ms=3, label="G' residual")
    ax.semilogx(omega, res_pp, 'r.-', ms=3, label="G'' residual")
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('(d) Fit Residuals vs True (noise-free)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Polymer Rheology Inversion (Rouse Model)  |  "
        f"PSNR = {metrics['psnr_dB']:.2f} dB  |  "
        f"CC = {metrics['correlation_coefficient']:.4f}  |  "
        f"Mean RE = {metrics['mean_parameter_relative_error']:.4f}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS]  Saved → {save_path}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  reptate_rheo — Polymer Rheology Inversion Pipeline")
    print("=" * 60)

    # --- 1. Generate synthetic data ---
    omega, G_p_obs, G_pp_obs, G_p_true, G_pp_true, true_params = generate_data()
    print(f"[DATA] {len(omega)} frequency points, "
          f"ω ∈ [{omega[0]:.2e}, {omega[-1]:.2e}] rad/s")
    print(f"[DATA] True params: G0={true_params['G0']:.2e} Pa, "
          f"τ_R={true_params['tau_R']:.4e} s, η_s={true_params['eta_s']:.1f} Pa·s")

    # --- 2. Inversion ---
    print("[RECON] Running differential evolution …")
    G_p_fit, G_pp_fit, fitted_params = reconstruct(omega, G_p_obs, G_pp_obs)
    print(f"[RECON] Fitted: G0={fitted_params['G0']:.2e}, "
          f"τ_R={fitted_params['tau_R']:.4e}, η_s={fitted_params['eta_s']:.2f}")

    # --- 3. Evaluate ---
    metrics = compute_metrics(
        true_params, fitted_params,
        G_p_true, G_pp_true,
        G_p_fit, G_pp_fit,
    )
    print(f"[EVAL] PSNR = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC   = {metrics['correlation_coefficient']:.6f}")
    print(f"[EVAL] Mean RE = {metrics['mean_parameter_relative_error']:.6f}")
    for k, v in metrics['parameters'].items():
        print(f"       {k:>6s}: true={v['true']:.4e}  fitted={v['fitted']:.4e}  RE={v['rel_error']:.6f}")

    # --- 4. Save metrics ---
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # --- 5. Save arrays ---
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"),
            np.column_stack([G_p_true, G_pp_true]))
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"),
            np.column_stack([G_p_fit, G_pp_fit]))
    print(f"[SAVE] ground_truth.npy, recon_output.npy → {RESULTS_DIR}")

    # --- 6. Visualise ---
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize(omega, G_p_obs, G_pp_obs, G_p_true, G_pp_true,
              G_p_fit, G_pp_fit, metrics, vis_path)

    print("=" * 60)
    print("  DONE")
    print("=" * 60)
