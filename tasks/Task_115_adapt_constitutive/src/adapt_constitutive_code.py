"""
adapt_constitutive — Constitutive Model Calibration
=====================================================
From experimental stress-strain data, calibrate constitutive model parameters
(elastic modulus, yield stress, hardening coefficient, hardening exponent) by
inverse optimisation using a power-law hardening model.

Physics:
  Forward:
    σ = E × ε                               for ε ≤ ε_y  (elastic)
    σ = σ_y + K × (ε − ε_y)^n              for ε > ε_y  (plastic, power-law)
    where ε_y = σ_y / E

  Inverse:
    scipy.optimize.minimize (L-BFGS-B) minimises ||σ_obs − σ_model(params)||²
    to recover [E, σ_y, K, n] from a noisy stress-strain curve.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import minimize

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_115_adapt_constitutive"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── ground-truth parameters ───────────────────────────────────────
E_TRUE   = 210e3    # Young's modulus (MPa)  — steel-like
SY_TRUE  = 350.0    # Yield stress (MPa)
K_TRUE   = 800.0    # Hardening coefficient (MPa)
N_TRUE   = 0.45     # Hardening exponent

# ── strain grid ───────────────────────────────────────────────────
STRAIN_MAX = 0.15   # 15 % total strain
N_POINTS   = 500
NOISE_LEVEL = 0.02  # 2 % Gaussian noise on stress
SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1.  FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════
def constitutive_model(strain, E, sigma_y, K, n):
    """
    Power-law hardening constitutive model.

    σ = E ε                          for ε ≤ ε_y
    σ = σ_y + K (ε − ε_y)^n         for ε >  ε_y
    """
    eps_y = sigma_y / E
    stress = np.empty_like(strain)
    elastic = strain <= eps_y
    plastic = ~elastic
    stress[elastic] = E * strain[elastic]
    eps_p = strain[plastic] - eps_y
    eps_p = np.maximum(eps_p, 0.0)
    stress[plastic] = sigma_y + K * np.power(eps_p, n)
    return stress


# ═══════════════════════════════════════════════════════════════════
# 2.  GENERATE GROUND-TRUTH (synthetic experiment)
# ═══════════════════════════════════════════════════════════════════
def generate_gt(strain):
    """Return clean + noisy stress for the GT parameter set."""
    stress_clean = constitutive_model(strain, E_TRUE, SY_TRUE, K_TRUE, N_TRUE)
    noise = NOISE_LEVEL * np.max(np.abs(stress_clean)) * np.random.randn(len(strain))
    stress_noisy = stress_clean + noise
    return stress_clean, stress_noisy


# ═══════════════════════════════════════════════════════════════════
# 3.  INVERSE:  L-BFGS-B OPTIMISATION
# ═══════════════════════════════════════════════════════════════════
def objective(params, strain, stress_obs):
    """Sum-of-squares misfit."""
    E, sigma_y, K, n = params
    stress_pred = constitutive_model(strain, E, sigma_y, K, n)
    return np.sum((stress_obs - stress_pred) ** 2)


def calibrate(strain, stress_obs):
    """Recover constitutive parameters from noisy stress-strain data."""
    # Estimate E from initial slope (first ~10% of data where elastic)
    n_init = max(5, len(strain) // 20)
    E_est = float(np.polyfit(strain[:n_init], stress_obs[:n_init], 1)[0])
    E_est = np.clip(E_est, 50e3, 500e3)

    # Estimate yield stress from the knee
    sy_est = float(stress_obs[n_init * 2]) if n_init * 2 < len(stress_obs) else 300.0

    x0 = [E_est, sy_est, 700.0, 0.4]
    bounds = [(50e3, 500e3),   # E
              (100.0, 800.0),  # σ_y
              (100.0, 2000.0), # K
              (0.05, 0.95)]    # n

    best_result = None
    best_cost = np.inf
    # Multi-start to avoid local minima
    starts = [
        x0,
        [E_est * 0.9, sy_est * 1.1, 800.0, 0.5],
        [E_est * 1.1, sy_est * 0.9, 600.0, 0.35],
        [200e3, 350.0, 800.0, 0.45],
    ]
    for s in starts:
        result = minimize(objective, s, args=(strain, stress_obs),
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 10000, "ftol": 1e-18, "gtol": 1e-14})
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
    return best_result.x


# ═══════════════════════════════════════════════════════════════════
# 4.  METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(stress_gt, stress_recon, params_true, params_fit):
    """PSNR, CC, parameter relative errors."""
    # PSNR on stress curve
    mse = np.mean((stress_gt - stress_recon) ** 2)
    data_range = np.max(stress_gt) - np.min(stress_gt)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))

    # Correlation coefficient
    cc = float(np.corrcoef(stress_gt.ravel(), stress_recon.ravel())[0, 1])

    # RMSE
    rmse = float(np.sqrt(mse))

    # Parameter relative errors
    names = ["E", "sigma_y", "K", "n"]
    param_errors = {}
    for name, pt, pf in zip(names, params_true, params_fit):
        re = abs(pf - pt) / abs(pt) * 100.0
        param_errors[f"{name}_true"] = float(pt)
        param_errors[f"{name}_fitted"] = float(pf)
        param_errors[f"{name}_RE_pct"] = float(re)

    metrics = {"PSNR": float(psnr), "CC": float(cc), "RMSE": float(rmse)}
    metrics.update(param_errors)
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════
def visualize(strain, stress_gt, stress_noisy, stress_recon, params_true, params_fit, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) GT vs Fitted stress-strain curve
    ax = axes[0]
    ax.plot(strain * 100, stress_gt, "k-", lw=2, label="Ground Truth")
    ax.plot(strain * 100, stress_noisy, ".", color="gray", ms=1, alpha=0.4, label="Noisy Observation")
    ax.plot(strain * 100, stress_recon, "r--", lw=2, label="Fitted Model")
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Stress-Strain Curve  (PSNR={metrics['PSNR']:.1f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Parameter comparison
    ax = axes[1]
    names = ["E (MPa)", "σ_y (MPa)", "K (MPa)", "n"]
    keys  = ["E", "sigma_y", "K", "n"]
    true_vals = [metrics[f"{k}_true"] for k in keys]
    fit_vals  = [metrics[f"{k}_fitted"] for k in keys]
    # Normalise for bar chart
    norm_true = [1.0] * 4
    norm_fit  = [f / t if t != 0 else 0 for f, t in zip(fit_vals, true_vals)]
    x = np.arange(4)
    ax.bar(x - 0.18, norm_true, 0.35, label="True", color="steelblue")
    ax.bar(x + 0.18, norm_fit,  0.35, label="Fitted", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Parameter Comparison (normalised)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Residual
    ax = axes[2]
    residual = stress_gt - stress_recon
    ax.plot(strain * 100, residual, "b-", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Residual Stress (MPa)")
    ax.set_title(f"Residual  (RMSE={metrics['RMSE']:.2f} MPa, CC={metrics['CC']:.4f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("adapt_constitutive — Constitutive Model Calibration")
    print("=" * 60)

    # 1. Strain grid
    strain = np.linspace(0, STRAIN_MAX, N_POINTS)

    # 2. Ground truth
    print("[1/4] Generating ground-truth stress-strain curve ...")
    stress_gt, stress_noisy = generate_gt(strain)

    # 3. Inverse calibration
    print("[2/4] Running L-BFGS-B inverse calibration ...")
    params_fit = calibrate(strain, stress_noisy)
    print(f"  Fitted: E={params_fit[0]:.1f}, σ_y={params_fit[1]:.2f}, "
          f"K={params_fit[2]:.2f}, n={params_fit[3]:.4f}")

    # 4. Reconstruct stress curve with fitted params
    stress_recon = constitutive_model(strain, *params_fit)

    # 5. Metrics
    params_true = [E_TRUE, SY_TRUE, K_TRUE, N_TRUE]
    metrics = compute_metrics(stress_gt, stress_recon, params_true, params_fit)
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.6f}")
    print(f"  RMSE = {metrics['RMSE']:.4f} MPa")
    for k in ["E", "sigma_y", "K", "n"]:
        print(f"  {k}: true={metrics[f'{k}_true']:.4f}  fit={metrics[f'{k}_fitted']:.4f}  RE={metrics[f'{k}_RE_pct']:.2f}%")

    # 6. Save
    print("[3/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), stress_gt)
        np.save(os.path.join(d, "recon_output.npy"), stress_recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # 7. Visualise
    print("[4/4] Generating visualisation ...")
    visualize(strain, stress_gt, stress_noisy, stress_recon, params_true, params_fit, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
