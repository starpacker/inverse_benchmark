"""
refnx_xrr — X-ray Reflectometry Inverse Problem
==================================================
Task: Recover thin-film layer structure (thickness, roughness, SLD)
      from measured X-ray reflectivity curve R(Q).

Inverse Problem:
    Given a reflectivity curve R(Q), recover the layer parameters
    (thickness d, roughness σ, SLD ρ) of a multilayer thin film.

Forward Model (Parratt/Abeles recursion via refnx):
    R(Q) = |r(Q)|²  where r is computed by refnx.reflect using
    the Abeles transfer-matrix formalism with Névot-Croce roughness.

Inverse Solver:
    Differential Evolution + Levenberg-Marquardt refinement
    using refnx.analysis.CurveFitter.

Repo: https://github.com/refnx/refnx
Paper: Nelson & Prescott (2019), J. Appl. Cryst., 52, 193.
       doi:10.1107/S1600576718017296

Metrics:
    - RMSE(log₁₀R): root-mean-square error in log₁₀(reflectivity) space
    - CC(log₁₀R): Pearson correlation of log₁₀(R) curves
    - Parameter errors: |Δd|, |ΔSLD|, |Δσ| for each fitted layer

Usage:
    /data/yjh/spectro_env/bin/python refnx_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

# ── refnx library imports ──────────────────────────────────
from refnx.reflect import SLD as SLDobj, ReflectModel, Structure
from refnx.dataset import ReflectDataset
from refnx.analysis import Objective, CurveFitter, Parameter

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Ground-truth multilayer definition ---
# Layer structure: air | polymer | SiO₂ | Si
# SLD in units of 1e-6 Å^-2
GT_PARAMS = {
    "polymer_thick": 250.0,    # Å
    "polymer_sld": 1.50,       # 1e-6 Å^-2
    "polymer_rough": 8.0,      # Å  (top roughness)
    "sio2_thick": 15.0,        # Å
    "sio2_sld": 18.88,         # 1e-6 Å^-2
    "sio2_rough": 3.0,         # Å
    "si_sld": 20.07,           # 1e-6 Å^-2
    "si_rough": 3.0,           # Å  (interface roughness)
    "bkg": 1e-7,               # instrumental background
}

Q_MIN = 0.005   # Å^-1
Q_MAX = 0.30    # Å^-1
N_POINTS = 400  # number of Q points
DQ_OVER_Q = 0.05   # 5 % dQ/Q resolution smearing
NOISE_FRAC = 0.03  # relative Poisson-like noise level
SEED = 42


# ═══════════════════════════════════════════════════════════
# 2. Build refnx Structure (used for both forward & inverse)
# ═══════════════════════════════════════════════════════════
def build_structure(params, vary=False):
    """
    Build a refnx Structure from a parameter dict.

    Parameters
    ----------
    params : dict
        Keys: polymer_thick, polymer_sld, polymer_rough,
              sio2_thick, sio2_sld, sio2_rough,
              si_sld, si_rough, bkg
    vary : bool
        If True, set bounds and mark parameters as variable
        (for fitting). If False, fix all parameters (for
        forward generation).

    Returns
    -------
    structure : refnx.reflect.Structure
    model : refnx.reflect.ReflectModel
    """
    air = SLDobj(0.0, name='air')
    polymer = SLDobj(params["polymer_sld"], name='polymer')
    sio2 = SLDobj(params["sio2_sld"], name='sio2')
    si = SLDobj(params["si_sld"], name='si')

    # air → polymer
    polymer_slab = polymer(params["polymer_thick"], params["polymer_rough"])
    # polymer → SiO2
    sio2_slab = sio2(params["sio2_thick"], params["sio2_rough"])
    # SiO2 → Si substrate
    si_slab = si(0, params["si_rough"])

    if vary:
        polymer_slab.thick.setp(bounds=(50, 500), vary=True)
        polymer_slab.sld.real.setp(bounds=(0.3, 6.0), vary=True)
        polymer_slab.rough.setp(bounds=(1, 25), vary=True)
        sio2_slab.thick.setp(bounds=(3, 50), vary=True)
        sio2_slab.rough.setp(bounds=(0.5, 15), vary=True)
        si_slab.rough.setp(bounds=(0.5, 15), vary=True)

    structure = air | polymer_slab | sio2_slab | si_slab
    model = ReflectModel(structure, bkg=params["bkg"], dq=DQ_OVER_Q * 100)

    if vary:
        model.bkg.setp(bounds=(1e-10, 1e-5), vary=True)

    return structure, model, {
        "polymer_slab": polymer_slab,
        "sio2_slab": sio2_slab,
        "si_slab": si_slab,
    }


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator (using refnx ReflectModel)
# ═══════════════════════════════════════════════════════════
def forward_operator(params, q):
    """
    Compute specular X-ray reflectivity R(Q) using the refnx Abeles
    matrix formalism.

    This is the ACTUAL refnx forward engine — not a hand-coded
    reimplementation.  refnx internally uses the Parratt (or Abeles)
    recursive algorithm with Névot-Croce interfacial roughness factors
    and optional resolution smearing.

    Parameters
    ----------
    params : dict   Ground-truth layer parameters.
    q : np.ndarray  Momentum-transfer values [Å^-1].

    Returns
    -------
    R : np.ndarray  Reflectivity curve.
    """
    _, model, _ = build_structure(params, vary=False)
    R = model(q)
    return R


# ═══════════════════════════════════════════════════════════
# 4. Data Generation (Synthetic Benchmark)
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """
    Generate synthetic XRR benchmark data.

    1. Build ground-truth structure with refnx.
    2. Compute clean reflectivity R(Q) via refnx forward.
    3. Add realistic Poisson-like noise + background.
    """
    print("[DATA] Building ground-truth multilayer with refnx ...")
    q = np.linspace(Q_MIN, Q_MAX, N_POINTS)

    R_clean = forward_operator(GT_PARAMS, q)
    print(f"[DATA] R(Q) range: [{R_clean.min():.3e}, {R_clean.max():.3e}]")

    # Realistic noise: relative Gaussian noise scaled by sqrt(R)
    rng = np.random.default_rng(SEED)
    sigma_R = np.maximum(R_clean * NOISE_FRAC, 1e-12)
    R_noisy = R_clean + sigma_R * rng.standard_normal(N_POINTS)
    R_noisy = np.maximum(R_noisy, 1e-12)  # reflectivity ≥ 0

    # Error bars for the fitter
    R_err = sigma_R

    print(f"[DATA] Added {NOISE_FRAC*100:.0f}% relative noise "
          f"({N_POINTS} points, Q ∈ [{Q_MIN}, {Q_MAX}] Å⁻¹)")

    return q, R_noisy, R_clean, R_err


# ═══════════════════════════════════════════════════════════
# 5. Inverse Solver (refnx CurveFitter)
# ═══════════════════════════════════════════════════════════
def reconstruct(q, R_meas, R_err):
    """
    Fit the XRR curve using refnx's Differential Evolution +
    Levenberg-Marquardt pipeline.

    The refnx CurveFitter wraps scipy.optimize.differential_evolution
    for global search, then scipy.optimize.least_squares for local
    refinement.  Resolution smearing (dq) is handled internally.

    Parameters
    ----------
    q : np.ndarray      Q values [Å^-1].
    R_meas : np.ndarray Measured reflectivity.
    R_err : np.ndarray  Error bars on reflectivity.

    Returns
    -------
    fitted_params : dict  Recovered layer parameters.
    R_fit : np.ndarray    Best-fit reflectivity curve.
    """
    # Build fitting model with free parameters
    initial_guess = {
        "polymer_thick": 200.0,   # deliberately off from GT
        "polymer_sld": 2.0,
        "polymer_rough": 5.0,
        "sio2_thick": 10.0,
        "sio2_sld": GT_PARAMS["sio2_sld"],  # fixed (known)
        "sio2_rough": 5.0,
        "si_sld": GT_PARAMS["si_sld"],       # fixed (known)
        "si_rough": 5.0,
        "bkg": 1e-7,
    }

    structure, model, slabs = build_structure(initial_guess, vary=True)

    # Create refnx dataset  (Q, R, dR, dQ)
    dq = q * DQ_OVER_Q
    dataset = ReflectDataset(data=(q, R_meas, R_err, dq))

    objective = Objective(model, dataset)

    # ── Stage 1: Differential Evolution (global) ──
    print("[RECON] Stage 1 — Differential Evolution (refnx.CurveFitter) ...")
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution', seed=SEED, maxiter=150, tol=1e-5)
    chi2_de = objective.chisqr()
    print(f"[RECON]   χ² after DE = {chi2_de:.4f}")

    # ── Stage 2: Least-Squares refinement (local) ──
    print("[RECON] Stage 2 — Least-Squares refinement ...")
    fitter.fit('least_squares')
    chi2_lm = objective.chisqr()
    print(f"[RECON]   χ² after LM = {chi2_lm:.4f}")

    # Extract fitted parameter values
    p = slabs["polymer_slab"]
    s = slabs["sio2_slab"]
    si = slabs["si_slab"]

    fitted_params = {
        "polymer_thick": float(p.thick.value),
        "polymer_sld": float(p.sld.real.value),
        "polymer_rough": float(p.rough.value),
        "sio2_thick": float(s.thick.value),
        "sio2_sld": float(s.sld.real.value),
        "sio2_rough": float(s.rough.value),
        "si_sld": GT_PARAMS["si_sld"],
        "si_rough": float(si.rough.value),
        "bkg": float(model.bkg.value),
    }

    R_fit = model(q, x_err=dq)

    return fitted_params, R_fit, chi2_lm


# ═══════════════════════════════════════════════════════════
# 6. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_params, fit_params, R_clean, R_fit, q):
    """
    Compute reconstruction quality metrics.

    Reflectivity-space:
      - RMSE(log₁₀R)
      - CC(log₁₀R): Pearson correlation in log₁₀ space
      - PSNR(log₁₀R)
      - SSIM(log₁₀R)

    Parameter-space:
      - Absolute errors for thickness, SLD, roughness of each layer
      - Relative errors
    """
    from skimage.metrics import structural_similarity as ssim_fn

    logR_gt = np.log10(np.maximum(R_clean, 1e-12))
    logR_fit = np.log10(np.maximum(R_fit, 1e-12))

    # Reflectivity metrics
    rmse_logR = float(np.sqrt(np.mean((logR_gt - logR_fit) ** 2)))
    cc_logR = float(np.corrcoef(logR_gt, logR_fit)[0, 1])

    data_range = logR_gt.max() - logR_gt.min()
    mse = np.mean((logR_gt - logR_fit) ** 2)
    psnr_logR = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # 1-D SSIM: tile to make 2D (7×N) so win_size=7 works
    tile_rows = 7
    a2d = np.tile(logR_gt, (tile_rows, 1))
    b2d = np.tile(logR_fit, (tile_rows, 1))
    ssim_logR = float(ssim_fn(
        a2d, b2d,
        data_range=data_range, win_size=7
    ))

    # Parameter recovery
    param_keys = [
        ("polymer_thick", "d_polymer [Å]"),
        ("polymer_sld", "SLD_polymer [1e-6 Å⁻²]"),
        ("polymer_rough", "σ_polymer [Å]"),
        ("sio2_thick", "d_SiO₂ [Å]"),
        ("sio2_rough", "σ_SiO₂ [Å]"),
        ("si_rough", "σ_Si [Å]"),
    ]

    param_errors = {}
    for key, label in param_keys:
        gt = gt_params[key]
        fit = fit_params[key]
        err = abs(gt - fit)
        rel = err / max(abs(gt), 1e-12) * 100
        param_errors[f"gt_{key}"] = float(gt)
        param_errors[f"fit_{key}"] = float(fit)
        param_errors[f"abs_err_{key}"] = float(err)
        param_errors[f"rel_err_{key}"] = float(rel)

    metrics = {
        "PSNR_logR": psnr_logR,
        "SSIM_logR": ssim_logR,
        "CC_logR": cc_logR,
        "RMSE_logR": rmse_logR,
        **param_errors,
    }
    return metrics


# ═══════════════════════════════════════════════════════════
# 7. Visualization
# ═══════════════════════════════════════════════════════════
def sld_profile_from_params(params, z_max=400, n_pts=500):
    """Compute SLD-vs-depth profile using refnx Structure."""
    structure, _, _ = build_structure(params, vary=False)
    z = np.linspace(0, z_max, n_pts)
    sld_profile = structure.sld_profile(z)
    return sld_profile[:, 0], sld_profile[:, 1]


def visualize_results(q, R_meas, R_clean, R_fit,
                      gt_params, fit_params, metrics, save_path):
    """
    Three-panel figure:
      1. Reflectivity curves (log scale) with residuals
      2. SLD-vs-depth profile: GT vs fit
      3. Bar chart of parameter recovery
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Reflectivity
    ax = axes[0, 0]
    ax.semilogy(q, R_clean, 'b-', lw=2, label='Ground Truth', zorder=3)
    ax.semilogy(q, R_meas, 'k.', ms=1.5, alpha=0.4, label='Noisy Data')
    ax.semilogy(q, R_fit, 'r--', lw=1.5, label='refnx Fit', zorder=2)
    ax.set_xlabel('Q [Å⁻¹]')
    ax.set_ylabel('Reflectivity R(Q)')
    ax.set_title('(a) X-ray Reflectivity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Residuals in log space
    ax = axes[0, 1]
    logR_gt = np.log10(np.maximum(R_clean, 1e-12))
    logR_fit = np.log10(np.maximum(R_fit, 1e-12))
    ax.plot(q, logR_gt - logR_fit, 'g-', lw=1)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Q [Å⁻¹]')
    ax.set_ylabel('Δ log₁₀R')
    ax.set_title(f'(b) Residuals  RMSE={metrics["RMSE_logR"]:.4f}')
    ax.grid(True, alpha=0.3)

    # (c) SLD profile
    ax = axes[1, 0]
    try:
        z_gt, sld_gt = sld_profile_from_params(gt_params)
        z_fit, sld_fit = sld_profile_from_params(fit_params)
        ax.plot(z_gt, sld_gt, 'b-', lw=2, label='Ground Truth')
        ax.plot(z_fit, sld_fit, 'r--', lw=2, label='refnx Fit')
    except Exception:
        # Fallback: simple step plot
        ax.text(0.5, 0.5, 'SLD profile unavailable',
                transform=ax.transAxes, ha='center')
    ax.set_xlabel('Depth [Å]')
    ax.set_ylabel('SLD [10⁻⁶ Å⁻²]')
    ax.set_title('(c) SLD Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Parameter bar chart
    ax = axes[1, 1]
    labels = ['d_poly', 'SLD_poly', 'σ_poly', 'd_SiO₂', 'σ_SiO₂', 'σ_Si']
    keys = ['polymer_thick', 'polymer_sld', 'polymer_rough',
            'sio2_thick', 'sio2_rough', 'si_rough']
    gt_vals = [gt_params[k] for k in keys]
    fit_vals = [fit_params[k] for k in keys]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, gt_vals, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_vals, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_title('(d) Parameter Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"refnx — X-ray Reflectometry Inversion\n"
        f"PSNR(logR)={metrics['PSNR_logR']:.1f} dB  |  "
        f"SSIM(logR)={metrics['SSIM_logR']:.4f}  |  "
        f"CC(logR)={metrics['CC_logR']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 8. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  refnx — X-ray Reflectometry Inverse Problem")
    print("=" * 65)

    # Step 1: Generate data with refnx forward model
    q, R_meas, R_clean, R_err = load_or_generate_data()

    # Step 2: Inverse with refnx DE + LM
    print("\n[RECON] Fitting with refnx.analysis.CurveFitter ...")
    fit_params, R_fit, chi2 = reconstruct(q, R_meas, R_err)

    # Step 3: Metrics
    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(GT_PARAMS, fit_params, R_clean, R_fit, q)
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")

    # Step 4: Save
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), R_fit)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), R_clean)
    np.save(os.path.join(RESULTS_DIR, "measurements.npy"), R_meas)

    # Step 5: Visualize
    visualize_results(q, R_meas, R_clean, R_fit,
                      GT_PARAMS, fit_params, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 65)
    print("  DONE — refnx XRR benchmark complete")
    print("=" * 65)
