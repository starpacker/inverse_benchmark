"""
batman — Exoplanet Transit Photometry Inverse Problem
======================================================
Task: Recover planet orbital/physical parameters from a transit light curve.

Inverse Problem:
    Given a time-series light curve F(t) showing a planetary transit dip,
    recover the planet-to-star radius ratio Rp/Rs, orbital inclination i,
    scaled semi-major axis a/Rs, and limb-darkening coefficients.

Forward Model (batman – BAsic Transit Model cAlculatioN):
    F(t) = 1 - δ(t;  Rp/Rs, a/Rs, i, u₁, u₂, period, t0, e, ω)
    Uses Mandel & Agol (2002) analytic formulae implemented in batman.

Inverse Solver:
    scipy.optimize.differential_evolution (global) +
    scipy.optimize.minimize (local Nelder-Mead refinement)

Repo: https://github.com/lkreidberg/batman
Paper: Kreidberg (2015), PASP, 127, 957, 1161–1165.
       doi:10.1086/683602

Usage:
    /data/yjh/spectro_env/bin/python batman_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ── batman library import ──────────────────────────────────
import batman

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ground-truth transit parameters (Hot Jupiter-like system)
GT_PARAMS = {
    "rp":    0.1,      # planet radius / stellar radius  (Rp/Rs)
    "a":     15.0,     # semi-major axis / stellar radius (a/Rs)
    "inc":   87.0,     # orbital inclination [degrees]
    "ecc":   0.0,      # eccentricity (circular orbit)
    "w":     90.0,     # argument of periastron [degrees]
    "t0":    0.0,      # mid-transit time [days]
    "per":   3.5,      # orbital period [days]
    "u1":    0.3,      # quadratic limb-darkening coeff 1
    "u2":    0.2,      # quadratic limb-darkening coeff 2
}

N_TIME = 500           # number of time points
T_SPAN = 0.25          # observation window around transit [days]
NOISE_PPM = 200        # photometric noise [ppm]
SEED = 42


# ═══════════════════════════════════════════════════════════
# 2. Forward Operator (batman transit model)
# ═══════════════════════════════════════════════════════════
def forward_operator(params, t):
    """
    Compute transit light curve F(t) using batman.

    batman implements the Mandel & Agol (2002) analytic model
    for planetary transit light curves, supporting:
      - Uniform, linear, quadratic, nonlinear limb darkening
      - Eccentric orbits
      - Secondary eclipses

    Parameters
    ----------
    params : dict  Transit parameters.
    t : np.ndarray Time array [days].

    Returns
    -------
    flux : np.ndarray  Normalised flux F(t).
    """
    bm_params = batman.TransitParams()
    bm_params.rp = params["rp"]
    bm_params.a = params["a"]
    bm_params.inc = params["inc"]
    bm_params.ecc = params["ecc"]
    bm_params.w = params["w"]
    bm_params.t0 = params["t0"]
    bm_params.per = params["per"]
    bm_params.u = [params["u1"], params["u2"]]
    bm_params.limb_dark = "quadratic"

    model = batman.TransitModel(bm_params, t)
    flux = model.light_curve(bm_params)
    return flux


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic transit light curve with batman."""
    print("[DATA] Generating synthetic transit light curve with batman ...")

    t = np.linspace(-T_SPAN, T_SPAN, N_TIME)

    flux_clean = forward_operator(GT_PARAMS, t)

    # Transit depth check
    depth = 1.0 - flux_clean.min()
    print(f"[DATA] Transit depth = {depth*1e6:.0f} ppm  "
          f"(Rp/Rs = {GT_PARAMS['rp']:.3f})")

    # Add Gaussian photometric noise
    rng = np.random.default_rng(SEED)
    sigma = NOISE_PPM * 1e-6  # convert ppm to relative flux
    flux_noisy = flux_clean + sigma * rng.standard_normal(N_TIME)
    flux_err = np.full(N_TIME, sigma)

    print(f"[DATA] Noise = {NOISE_PPM} ppm  |  {N_TIME} points  |  "
          f"T ∈ [{t[0]:.3f}, {t[-1]:.3f}] days")

    return t, flux_noisy, flux_clean, flux_err


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver
# ═══════════════════════════════════════════════════════════
def reconstruct(t, flux_meas, flux_err):
    """
    Fit transit parameters using DE + Nelder-Mead through batman forward.

    Free parameters: rp, a, inc, u1, u2
    Fixed parameters: t0, per, ecc, w  (assumed known from ephemeris)

    Parameters
    ----------
    t : np.ndarray        Time array.
    flux_meas : np.ndarray Measured (noisy) flux.
    flux_err : np.ndarray  Error bars.

    Returns
    -------
    fit_params : dict  Best-fit parameter values.
    flux_fit : np.ndarray  Best-fit light curve.
    """
    from scipy.optimize import differential_evolution, minimize

    # Fixed parameters
    fixed = {k: GT_PARAMS[k] for k in ["t0", "per", "ecc", "w"]}

    def chi2(x):
        rp, a, inc, u1, u2 = x
        params = {
            "rp": rp, "a": a, "inc": inc,
            "u1": u1, "u2": u2, **fixed
        }
        try:
            model_flux = forward_operator(params, t)
        except Exception:
            return 1e20
        return np.sum(((flux_meas - model_flux) / flux_err) ** 2)

    # Bounds for free parameters
    bounds = [
        (0.01, 0.3),    # rp  (Rp/Rs)
        (2.0, 50.0),    # a   (a/Rs)
        (70.0, 90.0),   # inc [deg]
        (0.0, 0.8),     # u1
        (-0.3, 0.6),    # u2
    ]

    # Stage 1: Differential Evolution
    print("[RECON] Stage 1 — Differential Evolution (global search) ...")
    result_de = differential_evolution(
        chi2, bounds, seed=SEED,
        maxiter=150, tol=1e-5, popsize=15,
        mutation=(0.5, 1.5), recombination=0.8
    )
    print(f"[RECON]   χ² = {result_de.fun:.2f}  "
          f"(reduced χ²/ν = {result_de.fun/N_TIME:.4f})")

    # Stage 2: Nelder-Mead local refinement
    print("[RECON] Stage 2 — Nelder-Mead refinement ...")
    result_nm = minimize(
        chi2, result_de.x, method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    print(f"[RECON]   χ² = {result_nm.fun:.2f}")

    rp, a, inc, u1, u2 = result_nm.x
    fit_params = {
        "rp": float(rp), "a": float(a), "inc": float(inc),
        "u1": float(u1), "u2": float(u2), **fixed
    }

    flux_fit = forward_operator(fit_params, t)
    return fit_params, flux_fit


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_params, fit_params, flux_clean, flux_fit, t):
    """
    Compute light-curve and parameter-recovery metrics.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    # Light-curve metrics
    residual = flux_clean - flux_fit
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    cc = float(np.corrcoef(flux_clean, flux_fit)[0, 1])

    data_range = flux_clean.max() - flux_clean.min()
    mse = np.mean(residual ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    tile_rows = 7
    a2d = np.tile(flux_clean, (tile_rows, 1))
    b2d = np.tile(flux_fit, (tile_rows, 1))
    ssim = float(ssim_fn(
        a2d, b2d,
        data_range=data_range, win_size=7
    ))

    # Relative error
    norm_gt = np.linalg.norm(flux_clean)
    re = float(np.linalg.norm(residual) / max(norm_gt, 1e-12))

    # Parameter recovery
    free_keys = ["rp", "a", "inc", "u1", "u2"]
    param_metrics = {}
    for k in free_keys:
        gt_v = gt_params[k]
        fit_v = fit_params[k]
        param_metrics[f"gt_{k}"] = float(gt_v)
        param_metrics[f"fit_{k}"] = float(fit_v)
        param_metrics[f"abs_err_{k}"] = float(abs(gt_v - fit_v))
        if abs(gt_v) > 1e-12:
            param_metrics[f"rel_err_{k}_pct"] = float(abs(gt_v - fit_v) / abs(gt_v) * 100)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim,
        "CC": cc,
        "RMSE": rmse,
        "RE": re,
        **param_metrics,
    }
    return metrics


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(t, flux_meas, flux_clean, flux_fit,
                      gt_params, fit_params, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Light curves
    ax = axes[0, 0]
    ax.plot(t * 24, flux_meas, 'k.', ms=1, alpha=0.3, label='Noisy data')
    ax.plot(t * 24, flux_clean, 'b-', lw=2, label='Ground truth')
    ax.plot(t * 24, flux_fit, 'r--', lw=1.5, label='batman fit')
    ax.set_xlabel('Time from mid-transit [hours]')
    ax.set_ylabel('Relative flux')
    ax.set_title('(a) Transit Light Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Residuals
    ax = axes[0, 1]
    residual_ppm = (flux_clean - flux_fit) * 1e6
    ax.plot(t * 24, residual_ppm, 'g-', lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Residual [ppm]')
    ax.set_title(f'(b) Residuals  RMSE={metrics["RMSE"]*1e6:.1f} ppm')
    ax.grid(True, alpha=0.3)

    # (c) Transit depth zoom
    ax = axes[1, 0]
    mask = np.abs(t * 24) < 2  # within ±2 hours
    ax.plot(t[mask] * 24, flux_clean[mask], 'b-', lw=2, label='GT')
    ax.plot(t[mask] * 24, flux_fit[mask], 'r--', lw=2, label='Fit')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Flux')
    ax.set_title('(c) Transit Detail (±2 hr)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Parameter bar chart
    ax = axes[1, 1]
    keys = ["rp", "a", "inc", "u1", "u2"]
    labels = ["Rp/Rs", "a/Rs", "inc [°]", "u₁", "u₂"]
    gt_vals = [gt_params[k] for k in keys]
    fit_vals = [fit_params[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, gt_vals, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_vals, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('(d) Parameter Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"batman — Transit Photometry Inversion\n"
        f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"CC={metrics['CC']:.4f}  |  RMSE={metrics['RMSE']*1e6:.1f} ppm",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  batman — Exoplanet Transit Photometry Inverse Problem")
    print("=" * 65)

    t, flux_meas, flux_clean, flux_err = load_or_generate_data()

    print("\n[RECON] Fitting transit parameters via batman forward model ...")
    fit_params, flux_fit = reconstruct(t, flux_meas, flux_err)

    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(GT_PARAMS, fit_params, flux_clean, flux_fit, t)
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), flux_fit)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), flux_clean)
    np.save(os.path.join(RESULTS_DIR, "measurements.npy"), flux_meas)

    visualize_results(t, flux_meas, flux_clean, flux_fit,
                      GT_PARAMS, fit_params, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 65)
    print("  DONE — batman transit photometry benchmark complete")
    print("=" * 65)
