"""
refellips — Spectroscopic Ellipsometry Inverse Problem
========================================================
Task: Extract thin-film optical constants (n, k, thickness) from
      ellipsometric Ψ/Δ measurements.

Inverse Problem:
    Given Ψ(λ) and Δ(λ) ellipsometric angles, recover the refractive
    index n(λ), extinction coefficient k(λ), and film thickness d.

Forward Model (refellips):
    Uses the transfer-matrix method for stratified media to compute
    Ψ and Δ from layer optical constants and thicknesses.

Inverse Solver:
    Differential Evolution + Least-Squares refinement using
    refellips/refnx analysis framework.

Repo: https://github.com/refnx/refellips
Paper: Lesina et al. (2022), SoftwareX, 20, 101225.

Usage:
    /data/yjh/spectro_env/bin/python refellips_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import differential_evolution, minimize
from skimage.metrics import structural_similarity as ssim_fn

# ── refellips library imports ──────────────────────────────
try:
    import refellips
    from refellips import Cauchy, SLD_from_ellipsometry
    HAS_REFELLIPS = True
except ImportError:
    HAS_REFELLIPS = False

from refnx.reflect import SLD as SLDobj
from refnx.analysis import Objective, CurveFitter

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ground truth: Cauchy thin film on Si substrate
GT_PARAMS = {
    "thickness": 120.0,    # nm
    "A": 1.50,             # Cauchy coefficient A (n ≈ A + B/λ² + C/λ⁴)
    "B": 0.005,            # Cauchy coefficient B [µm²]
    "C": 0.0001,           # Cauchy coefficient C [µm⁴]
    "k_amp": 0.01,         # extinction coefficient amplitude
}

# Si substrate optical constants (from Palik)
N_SI_633 = 3.882      # n at 633 nm
K_SI_633 = 0.019      # k at 633 nm

WAVELENGTH_MIN = 300   # nm
WAVELENGTH_MAX = 900   # nm
N_WAVELENGTHS = 100
ANGLE_OF_INCIDENCE = 70.0  # degrees (Brewster angle region)
NOISE_LEVEL = 0.005   # noise on Ψ,Δ (degrees)
SEED = 42


# ═══════════════════════════════════════════════════════════
# 2. Forward Model (Transfer Matrix)
# ═══════════════════════════════════════════════════════════
def cauchy_n(wavelength_nm, A, B, C):
    """Cauchy dispersion: n(λ) = A + B/λ² + C/λ⁴"""
    lam_um = wavelength_nm / 1000.0
    return A + B / lam_um**2 + C / lam_um**4

def si_optical_constants(wavelength_nm):
    """Approximate Si optical constants (simple dispersion)."""
    # Simplified model based on Palik data
    lam_um = wavelength_nm / 1000.0
    n = N_SI_633 + 0.8 * (0.633 / lam_um - 1)
    k = K_SI_633 * (0.633 / lam_um) ** 2
    return n, k

def forward_operator(params, wavelengths, angle_deg):
    """
    Compute ellipsometric Ψ and Δ for a thin film on Si substrate
    using the transfer matrix method.

    This implements the standard 2×2 transfer matrix for
    ambient/film/substrate with Fresnel coefficients.

    Parameters
    ----------
    params : dict       Film parameters (thickness, Cauchy coeffs).
    wavelengths : array Wavelengths [nm].
    angle_deg : float   Angle of incidence [degrees].

    Returns
    -------
    psi : array  Ψ [degrees]
    delta : array  Δ [degrees]
    """
    theta0 = np.radians(angle_deg)
    n0 = 1.0  # air

    psi = np.zeros(len(wavelengths))
    delta = np.zeros(len(wavelengths))

    for i, lam in enumerate(wavelengths):
        # Film optical constants
        n1 = cauchy_n(lam, params["A"], params["B"], params["C"])
        k1 = params["k_amp"] * (400.0 / lam) ** 2  # Urbach-like absorption
        N1 = n1 + 1j * k1

        # Substrate
        n2, k2 = si_optical_constants(lam)
        N2 = n2 + 1j * k2

        # Snell's law
        sin_theta0 = np.sin(theta0)
        cos_theta0 = np.cos(theta0)
        cos_theta1 = np.sqrt(1 - (n0 * sin_theta0 / N1) ** 2)
        cos_theta2 = np.sqrt(1 - (n0 * sin_theta0 / N2) ** 2)

        # Fresnel coefficients: air→film
        rp01 = (N1 * cos_theta0 - n0 * cos_theta1) / (N1 * cos_theta0 + n0 * cos_theta1)
        rs01 = (n0 * cos_theta0 - N1 * cos_theta1) / (n0 * cos_theta0 + N1 * cos_theta1)

        # Fresnel coefficients: film→substrate
        rp12 = (N2 * cos_theta1 - N1 * cos_theta2) / (N2 * cos_theta1 + N1 * cos_theta2)
        rs12 = (N1 * cos_theta1 - N2 * cos_theta2) / (N1 * cos_theta1 + N2 * cos_theta2)

        # Phase thickness
        beta = 2 * np.pi * params["thickness"] * N1 * cos_theta1 / lam

        # Total reflection coefficients (Airy formula)
        phase = np.exp(-2j * beta)
        Rp = (rp01 + rp12 * phase) / (1 + rp01 * rp12 * phase)
        Rs = (rs01 + rs12 * phase) / (1 + rs01 * rs12 * phase)

        # Ellipsometric ratio
        rho = Rp / Rs
        psi[i] = np.degrees(np.arctan(np.abs(rho)))
        delta[i] = np.degrees(np.angle(rho))

    return psi, delta


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic ellipsometry data."""
    print("[DATA] Generating synthetic ellipsometry data ...")

    wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, N_WAVELENGTHS)
    psi_clean, delta_clean = forward_operator(GT_PARAMS, wavelengths, ANGLE_OF_INCIDENCE)

    rng = np.random.default_rng(SEED)
    psi_noisy = psi_clean + NOISE_LEVEL * rng.standard_normal(N_WAVELENGTHS)
    delta_noisy = delta_clean + NOISE_LEVEL * rng.standard_normal(N_WAVELENGTHS)

    print(f"[DATA] Ψ range: [{psi_clean.min():.2f}, {psi_clean.max():.2f}]°")
    print(f"[DATA] Δ range: [{delta_clean.min():.2f}, {delta_clean.max():.2f}]°")
    print(f"[DATA] {N_WAVELENGTHS} wavelengths, θ={ANGLE_OF_INCIDENCE}°")

    return wavelengths, psi_noisy, delta_noisy, psi_clean, delta_clean


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver
# ═══════════════════════════════════════════════════════════
def reconstruct(wavelengths, psi_meas, delta_meas):
    """
    Fit ellipsometric parameters using DE + Nelder-Mead.

    Free parameters: thickness, A, B, C, k_amp

    Returns
    -------
    fit_params : dict
    psi_fit, delta_fit : arrays
    """
    def cost(x):
        thick, A, B, C, k_amp = x
        params = {"thickness": thick, "A": A, "B": B, "C": C, "k_amp": k_amp}
        try:
            psi_calc, delta_calc = forward_operator(params, wavelengths, ANGLE_OF_INCIDENCE)
            res_psi = (psi_meas - psi_calc) / NOISE_LEVEL
            res_delta = (delta_meas - delta_calc) / NOISE_LEVEL
            return np.sum(res_psi**2 + res_delta**2)
        except Exception:
            return 1e20

    bounds = [
        (10, 500),       # thickness [nm]
        (1.3, 2.0),      # A
        (0.0, 0.05),     # B
        (0.0, 0.005),    # C
        (0.0, 0.1),      # k_amp
    ]

    print("[RECON] Stage 1 — Differential Evolution ...")
    result_de = differential_evolution(cost, bounds, seed=SEED, maxiter=150,
                                        tol=1e-5, popsize=15)
    print(f"[RECON]   χ² = {result_de.fun:.2f}")

    print("[RECON] Stage 2 — Nelder-Mead refinement ...")
    result = minimize(cost, result_de.x, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-6})
    print(f"[RECON]   χ² = {result.fun:.2f}")

    thick, A, B, C, k_amp = result.x
    fit_params = {"thickness": float(thick), "A": float(A), "B": float(B),
                  "C": float(C), "k_amp": float(k_amp)}
    psi_fit, delta_fit = forward_operator(fit_params, wavelengths, ANGLE_OF_INCIDENCE)

    return fit_params, psi_fit, delta_fit


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt, fit, psi_clean, psi_fit, delta_clean, delta_fit, wavelengths):
    """Compute ellipsometry reconstruction metrics."""
    # Ψ metrics
    rmse_psi = float(np.sqrt(np.mean((psi_clean - psi_fit)**2)))
    cc_psi = float(np.corrcoef(psi_clean, psi_fit)[0, 1])

    # Δ metrics
    rmse_delta = float(np.sqrt(np.mean((delta_clean - delta_fit)**2)))
    cc_delta = float(np.corrcoef(delta_clean, delta_fit)[0, 1])

    # Combined PSNR/SSIM on Ψ
    dr = psi_clean.max() - psi_clean.min()
    mse = np.mean((psi_clean - psi_fit)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    tile_rows = 7
    a2d = np.tile(psi_clean, (tile_rows, 1))
    b2d = np.tile(psi_fit, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))

    # Parameter recovery
    param_metrics = {}
    for k in ["thickness", "A", "B", "C", "k_amp"]:
        g, f = gt[k], fit[k]
        param_metrics[f"gt_{k}"] = float(g)
        param_metrics[f"fit_{k}"] = float(f)
        param_metrics[f"abs_err_{k}"] = float(abs(g - f))

    # n(λ) recovery
    n_gt = cauchy_n(wavelengths, gt["A"], gt["B"], gt["C"])
    n_fit = cauchy_n(wavelengths, fit["A"], fit["B"], fit["C"])
    cc_n = float(np.corrcoef(n_gt, n_fit)[0, 1])

    return {
        "PSNR_psi": psnr,
        "SSIM_psi": ssim_val,
        "CC_psi": cc_psi,
        "RMSE_psi_deg": rmse_psi,
        "CC_delta": cc_delta,
        "RMSE_delta_deg": rmse_delta,
        "CC_n": cc_n,
        **param_metrics,
    }


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(wavelengths, psi_meas, delta_meas, psi_clean, delta_clean,
                      psi_fit, delta_fit, gt, fit, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(wavelengths, psi_clean, 'b-', lw=2, label='GT')
    ax.plot(wavelengths, psi_meas, 'k.', ms=2, alpha=0.3, label='Noisy')
    ax.plot(wavelengths, psi_fit, 'r--', lw=1.5, label='Fit')
    ax.set_xlabel('Wavelength [nm]'); ax.set_ylabel('Ψ [°]')
    ax.set_title('(a) Ψ(λ)'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(wavelengths, delta_clean, 'b-', lw=2, label='GT')
    ax.plot(wavelengths, delta_meas, 'k.', ms=2, alpha=0.3, label='Noisy')
    ax.plot(wavelengths, delta_fit, 'r--', lw=1.5, label='Fit')
    ax.set_xlabel('Wavelength [nm]'); ax.set_ylabel('Δ [°]')
    ax.set_title('(b) Δ(λ)'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    n_gt = cauchy_n(wavelengths, gt["A"], gt["B"], gt["C"])
    n_fit = cauchy_n(wavelengths, fit["A"], fit["B"], fit["C"])
    ax.plot(wavelengths, n_gt, 'b-', lw=2, label='GT n(λ)')
    ax.plot(wavelengths, n_fit, 'r--', lw=2, label='Fit n(λ)')
    ax.set_xlabel('Wavelength [nm]'); ax.set_ylabel('Refractive index n')
    ax.set_title('(c) Dispersion'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = ['d [nm]', 'A', 'B', 'C', 'k_amp']
    keys = ['thickness', 'A', 'B', 'C', 'k_amp']
    gt_v = [gt[k] for k in keys]; fit_v = [fit[k] for k in keys]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, gt_v, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_v, w, label='Fit', color='tomato')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('(d) Parameters'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"refellips — Spectroscopic Ellipsometry Inversion\n"
                 f"PSNR(Ψ)={metrics['PSNR_psi']:.1f} dB  |  CC(Ψ)={metrics['CC_psi']:.4f}  |  "
                 f"CC(Δ)={metrics['CC_delta']:.4f}  |  Δd={metrics['abs_err_thickness']:.2f} nm",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  refellips — Spectroscopic Ellipsometry Inversion")
    print("=" * 65)
    wavelengths, psi_meas, delta_meas, psi_clean, delta_clean = load_or_generate_data()
    fit_params, psi_fit, delta_fit = reconstruct(wavelengths, psi_meas, delta_meas)
    metrics = compute_metrics(GT_PARAMS, fit_params, psi_clean, psi_fit,
                              delta_clean, delta_fit, wavelengths)
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), np.column_stack([psi_fit, delta_fit]))
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), np.column_stack([psi_clean, delta_clean]))
    visualize_results(wavelengths, psi_meas, delta_meas, psi_clean, delta_clean,
                      psi_fit, delta_fit, GT_PARAMS, fit_params, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)
