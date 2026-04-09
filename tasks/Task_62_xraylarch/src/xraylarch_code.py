"""
xraylarch — EXAFS Fitting Inverse Problem
============================================
Task: Extract local atomic structure from X-ray absorption fine
      structure (EXAFS) oscillations.

Inverse Problem:
    Given χ(k) EXAFS oscillations, recover structural parameters:
    coordination number N, bond distance R, Debye-Waller factor σ²,
    and energy shift ΔE₀ for each scattering shell.

Forward Model (xraylarch / FEFF theory):
    χ(k) = Σ_j (N_j S₀² |f_j(k)|) / (k R_j²) ·
            sin(2kR_j + δ_j(k)) · exp(-2σ_j²k²) · exp(-2R_j/λ(k))
    Standard EXAFS equation with backscattering amplitude f(k),
    phase shift δ(k), and mean-free-path λ(k).

Inverse Solver:
    Nonlinear least-squares fitting (Levenberg-Marquardt)
    using lmfit with Fourier-filtered back-transform.

Repo: https://github.com/xraypy/xraylarch
Paper: Newville (2013), J. Phys.: Conf. Ser., 430, 012007.

Usage:
    /data/yjh/spectro_env/bin/python xraylarch_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import least_squares, differential_evolution
from skimage.metrics import structural_similarity as ssim_fn

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ground truth: Fe-O first shell (iron oxide)
GT_SHELLS = [
    {"N": 6.0, "R": 1.98, "sigma2": 0.005, "dE0": 2.0,
     "label": "Fe-O", "Z": 8},
    {"N": 2.0, "R": 3.05, "sigma2": 0.008, "dE0": 2.0,
     "label": "Fe-Fe", "Z": 26},
]

S02 = 0.9           # amplitude reduction factor
K_MIN = 2.0         # Å^-1
K_MAX = 14.0        # Å^-1
N_K = 300           # k-space points
K_WEIGHT = 2        # k-weighting for fitting
NOISE_LEVEL = 0.005 # noise in χ(k)
SEED = 42

# FEFF-like scattering parameters (simplified)
def feff_amplitude(k, Z):
    """Simplified backscattering amplitude |f(k)|."""
    if Z == 8:  # O
        return 0.5 * np.exp(-0.01 * k**2) * (1 + 0.1 * np.sin(k))
    elif Z == 26:  # Fe
        return 0.8 * np.exp(-0.005 * k**2) * (1 + 0.2 * np.sin(1.5 * k))
    else:
        return 0.6 * np.exp(-0.008 * k**2)

def feff_phase(k, Z):
    """Simplified total phase shift δ(k)."""
    if Z == 8:
        return -0.2 * k + 0.5 + 0.02 * k**2
    elif Z == 26:
        return -0.3 * k + 1.0 + 0.015 * k**2
    else:
        return -0.25 * k + 0.7

def mean_free_path(k):
    """Mean free path λ(k) in Å."""
    return 1.0 / (0.003 * k**2 + 0.01)


# ═══════════════════════════════════════════════════════════
# 2. Forward Operator (EXAFS equation)
# ═══════════════════════════════════════════════════════════
def forward_operator(shells, k, s02=S02):
    """
    Compute EXAFS χ(k) from shell parameters.

    Standard EXAFS equation:
    χ(k) = Σ_j (N_j·S₀²·|f_j(k)|) / (k·R_j²) ·
            sin(2kR_j + δ_j(k)) ·
            exp(-2σ²_j·k²) · exp(-2R_j/λ(k))

    Parameters
    ----------
    shells : list of dict  Shell parameters.
    k : array              Photoelectron wavenumber [Å^-1].
    s02 : float            Amplitude reduction factor.

    Returns
    -------
    chi : array            EXAFS oscillation function.
    """
    chi = np.zeros_like(k)
    lam = mean_free_path(k)

    for sh in shells:
        N = sh["N"]
        R = sh["R"]
        sig2 = sh["sigma2"]
        dE0 = sh.get("dE0", 0)
        Z = sh["Z"]

        # Effective k with energy shift
        k_eff = k  # simplified; in real FEFF, k depends on E0

        amp = feff_amplitude(k_eff, Z)
        phase = feff_phase(k_eff, Z)

        chi += (N * s02 * amp / (k * R**2) *
                np.sin(2 * k * R + phase + 2 * k * dE0 * 0.01) *
                np.exp(-2 * sig2 * k**2) *
                np.exp(-2 * R / lam))

    return chi


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic EXAFS data."""
    print("[DATA] Generating synthetic EXAFS data (Fe-O/Fe-Fe) ...")
    k = np.linspace(K_MIN, K_MAX, N_K)
    chi_clean = forward_operator(GT_SHELLS, k)

    rng = np.random.default_rng(SEED)
    chi_noisy = chi_clean + NOISE_LEVEL * rng.standard_normal(N_K)

    print(f"[DATA] k range: [{K_MIN}, {K_MAX}] Å⁻¹, {N_K} points")
    print(f"[DATA] χ range: [{chi_clean.min():.4f}, {chi_clean.max():.4f}]")
    print(f"[DATA] Shells: {[s['label'] for s in GT_SHELLS]}")

    return k, chi_noisy, chi_clean


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver
# ═══════════════════════════════════════════════════════════
def reconstruct(k, chi_meas):
    """
    Fit EXAFS data to recover shell parameters.

    Uses DE + least_squares with FEFF-like forward model.
    """
    def residual(params):
        shells = [
            {"N": params[0], "R": params[1], "sigma2": params[2],
             "dE0": params[3], "Z": 8, "label": "Fe-O"},
            {"N": params[4], "R": params[5], "sigma2": params[6],
             "dE0": params[7], "Z": 26, "label": "Fe-Fe"},
        ]
        chi_calc = forward_operator(shells, k)
        return (chi_meas - chi_calc) * k**K_WEIGHT

    bounds_de = [
        (1, 12), (1.5, 2.5), (0.001, 0.02), (-5, 10),
        (0.5, 6), (2.5, 3.8), (0.001, 0.02), (-5, 10),
    ]

    print("[RECON] Stage 1 — Differential Evolution ...")
    result_de = differential_evolution(
        lambda p: np.sum(residual(p)**2), bounds_de,
        seed=SEED, maxiter=150, tol=1e-5
    )
    print(f"[RECON]   χ² = {result_de.fun:.6f}")

    print("[RECON] Stage 2 — Levenberg-Marquardt ...")
    lb = [b[0] for b in bounds_de]
    ub = [b[1] for b in bounds_de]
    result = least_squares(residual, result_de.x, bounds=(lb, ub),
                           method='trf', ftol=1e-8, xtol=1e-8)
    print(f"[RECON]   cost = {result.cost:.6f}")

    p = result.x
    fit_shells = [
        {"N": float(p[0]), "R": float(p[1]), "sigma2": float(p[2]),
         "dE0": float(p[3]), "Z": 8, "label": "Fe-O"},
        {"N": float(p[4]), "R": float(p[5]), "sigma2": float(p[6]),
         "dE0": float(p[7]), "Z": 26, "label": "Fe-Fe"},
    ]
    chi_fit = forward_operator(fit_shells, k)

    return fit_shells, chi_fit


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_shells, fit_shells, chi_clean, chi_fit, k):
    """Compute EXAFS fitting metrics."""
    # k-weighted χ
    kw_gt = chi_clean * k**K_WEIGHT
    kw_fit = chi_fit * k**K_WEIGHT

    cc = float(np.corrcoef(kw_gt, kw_fit)[0, 1])
    rmse = float(np.sqrt(np.mean((kw_gt - kw_fit)**2)))
    dr = kw_gt.max() - kw_gt.min()
    mse = np.mean((kw_gt - kw_fit)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    # 1-D SSIM: tile to make 2D (7×N) so win_size=7 works
    tile_rows = 7
    a2d = np.tile(kw_gt, (tile_rows, 1))
    b2d = np.tile(kw_fit, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))
    re = float(np.linalg.norm(kw_gt - kw_fit) / max(np.linalg.norm(kw_gt), 1e-12))

    # R-space (FT) comparison
    window = np.hanning(len(k))
    ft_gt = np.abs(np.fft.fft(kw_gt * window))[:len(k)//2]
    ft_fit = np.abs(np.fft.fft(kw_fit * window))[:len(k)//2]
    cc_ft = float(np.corrcoef(ft_gt, ft_fit)[0, 1])

    # Parameter recovery
    param_metrics = {}
    for i, (gt_sh, fit_sh) in enumerate(zip(gt_shells, fit_shells)):
        for key in ["N", "R", "sigma2"]:
            g, f = gt_sh[key], fit_sh[key]
            param_metrics[f"gt_{gt_sh['label']}_{key}"] = float(g)
            param_metrics[f"fit_{gt_sh['label']}_{key}"] = float(f)
            param_metrics[f"err_{gt_sh['label']}_{key}"] = float(abs(g-f))

    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse,
            "CC_FT": cc_ft, **param_metrics}


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(k, chi_meas, chi_clean, chi_fit, gt_shells, fit_shells,
                      metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) k²χ(k)
    axes[0, 0].plot(k, chi_clean * k**2, 'b-', lw=2, label='GT')
    axes[0, 0].plot(k, chi_meas * k**2, 'k.', ms=1, alpha=0.3, label='Noisy')
    axes[0, 0].plot(k, chi_fit * k**2, 'r--', lw=1.5, label='Fit')
    axes[0, 0].set_xlabel('k [Å⁻¹]'); axes[0, 0].set_ylabel('k²χ(k)')
    axes[0, 0].set_title('(a) EXAFS'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # (b) Fourier transform (R-space)
    window = np.hanning(len(k))
    r = np.fft.fftfreq(len(k), d=(k[1]-k[0])/(2*np.pi))[:len(k)//2]
    ft_gt = np.abs(np.fft.fft(chi_clean * k**2 * window))[:len(k)//2]
    ft_fit = np.abs(np.fft.fft(chi_fit * k**2 * window))[:len(k)//2]
    axes[0, 1].plot(r, ft_gt, 'b-', lw=2, label='GT')
    axes[0, 1].plot(r, ft_fit, 'r--', lw=1.5, label='Fit')
    axes[0, 1].set_xlabel('R [Å]'); axes[0, 1].set_ylabel('|FT|')
    axes[0, 1].set_title('(b) Radial Distribution'); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_xlim(0, 5)

    # (c) Residual
    axes[1, 0].plot(k, (chi_clean - chi_fit) * k**2, 'g-', lw=1)
    axes[1, 0].axhline(0, color='k', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('k [Å⁻¹]'); axes[1, 0].set_ylabel('Residual k²Δχ')
    axes[1, 0].set_title(f'(c) Residual  RMSE={metrics["RMSE"]:.4f}')
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Parameter bars
    labels, gt_v, fit_v = [], [], []
    for gs, fs in zip(gt_shells, fit_shells):
        for key in ["N", "R", "sigma2"]:
            labels.append(f"{gs['label']}_{key}")
            gt_v.append(gs[key]); fit_v.append(fs[key])
    x = np.arange(len(labels)); w = 0.35
    axes[1, 1].bar(x - w/2, gt_v, w, label='GT', color='steelblue')
    axes[1, 1].bar(x + w/2, fit_v, w, label='Fit', color='tomato')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(labels, fontsize=7, rotation=30)
    axes[1, 1].set_title('(d) Parameters'); axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"xraylarch — EXAFS Fitting\nPSNR={metrics['PSNR']:.1f} dB  |  "
                 f"SSIM={metrics['SSIM']:.4f}  |  CC={metrics['CC']:.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[VIS] Saved → {save_path}")


if __name__ == "__main__":
    print("=" * 65)
    print("  xraylarch — EXAFS Fitting")
    print("=" * 65)
    k, chi_meas, chi_clean = load_or_generate_data()
    fit_shells, chi_fit = reconstruct(k, chi_meas)
    metrics = compute_metrics(GT_SHELLS, fit_shells, chi_clean, chi_fit, k)
    for key, val in sorted(metrics.items()):
        print(f"  {key:30s} = {val}")
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), chi_fit)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), chi_clean)
    visualize_results(k, chi_meas, chi_clean, chi_fit, GT_SHELLS, fit_shells,
                      metrics, os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)
