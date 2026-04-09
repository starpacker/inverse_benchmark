"""
sme_stellar - Stellar Spectral Synthesis & Parameter Fitting
=============================================================
From a high-resolution stellar spectrum, fit stellar parameters
(T_eff, log g, [Fe/H]) and element abundances.

Physics:
  - Forward model: Planck continuum × Voigt absorption lines
  - 5 elements (Fe, Ca, Mg, Na, Ti), each with 2–3 characteristic lines
  - Line depth depends on abundance, width on T_eff / log_g
  - Inverse: scipy.optimize.differential_evolution
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.special import voigt_profile
from scipy.optimize import differential_evolution

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_100_sme_stellar"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── physical parameters ───────────────────────────────────────────
# Wavelength grid (Angstroms) — high-res optical range
WAVE_MIN   = 5000.0
WAVE_MAX   = 6500.0
N_WAVE     = 2000
WAVELENGTH = np.linspace(WAVE_MIN, WAVE_MAX, N_WAVE)

# Ground-truth stellar parameters
GT_TEFF    = 5780.0     # K  (solar-like)
GT_LOGG    = 4.44       # log(cm/s^2)
GT_FEH     = 0.0        # [Fe/H] dex

# Element line list: (element, rest_wavelength_AA, gf_value)
# Each element has 2–3 characteristic absorption lines
LINE_LIST = [
    ("Fe", 5171.0, -1.79), ("Fe", 5270.0, -1.34), ("Fe", 5328.0, -1.47),
    ("Ca", 5261.0, -0.58), ("Ca", 5588.0, -0.22),
    ("Mg", 5172.7, -0.39), ("Mg", 5183.6, -0.17), ("Mg", 5528.4, -0.62),
    ("Na", 5890.0, 0.11),  ("Na", 5896.0, -0.18),
    ("Ti", 5173.7, -1.06), ("Ti", 5336.8, -1.59),
]

# Ground-truth abundances [X/H] in dex (solar = 0)
GT_ABUNDANCES = {"Fe": 0.0, "Ca": 0.05, "Mg": -0.1, "Na": 0.15, "Ti": -0.05}

SNR = 80.0   # signal-to-noise ratio

np.random.seed(42)


# ====================================================================
# 1. Forward model
# ====================================================================
def planck_continuum(wavelength_AA, T_eff):
    """Planck function as continuum (normalised, in wavelength units)."""
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    lam_m = wavelength_AA * 1e-10
    B = (2.0 * h * c**2 / lam_m**5) / (np.exp(h * c / (lam_m * k * T_eff)) - 1.0)
    return B / B.max()


def voigt_line(wavelength, lam0, sigma_G, gamma_L, depth):
    """Single Voigt absorption line."""
    delta = wavelength - lam0
    profile = voigt_profile(delta, sigma_G, gamma_L)
    profile = profile / profile.max() if profile.max() > 0 else profile
    return depth * profile


def synthesize_spectrum(wavelength, T_eff, log_g, feh, abundances):
    """
    Synthesize a stellar spectrum.
    
    Parameters
    ----------
    wavelength : array, Angstroms
    T_eff : float, K
    log_g : float
    feh  : float, [Fe/H]
    abundances : dict  {element: [X/H]}
    
    Returns
    -------
    flux : array, normalised flux
    """
    continuum = planck_continuum(wavelength, T_eff)
    absorption = np.zeros_like(wavelength)

    for elem, lam0, gf in LINE_LIST:
        ab = abundances.get(elem, 0.0) + feh
        # Gaussian width depends on T_eff (thermal broadening)
        sigma_thermal = lam0 / 3e5 * np.sqrt(2.0 * 1.381e-23 * T_eff / (56.0 * 1.66e-27))
        sigma_G = max(sigma_thermal, 0.02)
        # Lorentzian width depends on log_g (pressure broadening)
        gamma_L = 0.05 * 10.0**(0.3 * (4.5 - log_g))
        # Line depth depends on abundance and oscillator strength
        depth = 0.3 * 10.0**(gf + ab) / (1.0 + 10.0**(gf + ab))
        depth = np.clip(depth, 0.0, 0.95)
        absorption += voigt_line(wavelength, lam0, sigma_G, gamma_L, depth)

    flux = continuum * (1.0 - np.clip(absorption, 0, 0.99))
    return flux


# ====================================================================
# 2. Generate ground-truth data
# ====================================================================
def generate_gt_data():
    """Generate GT spectrum + noisy observation."""
    flux_gt = synthesize_spectrum(WAVELENGTH, GT_TEFF, GT_LOGG, GT_FEH, GT_ABUNDANCES)
    noise_level = flux_gt.mean() / SNR
    noise = np.random.normal(0, noise_level, N_WAVE)
    flux_obs = flux_gt + noise
    return flux_gt, flux_obs


# ====================================================================
# 3. Inverse: differential evolution
# ====================================================================
def pack_params(T_eff, log_g, feh, abundances):
    """Pack parameters into a 1D vector."""
    vec = [T_eff, log_g, feh]
    for elem in sorted(abundances.keys()):
        vec.append(abundances[elem])
    return np.array(vec)


def unpack_params(vec):
    """Unpack 1D vector into parameters."""
    T_eff = vec[0]
    log_g = vec[1]
    feh   = vec[2]
    elements = sorted(GT_ABUNDANCES.keys())
    abundances = {elem: vec[3 + i] for i, elem in enumerate(elements)}
    return T_eff, log_g, feh, abundances


def objective(vec, wavelength, flux_obs):
    """Chi-squared objective."""
    T_eff, log_g, feh, abundances = unpack_params(vec)
    flux_model = synthesize_spectrum(wavelength, T_eff, log_g, feh, abundances)
    residual = flux_obs - flux_model
    return np.sum(residual**2)


def solve_inverse(wavelength, flux_obs):
    """Fit stellar parameters using differential evolution."""
    bounds = [
        (4500, 7000),   # T_eff
        (3.0, 5.5),     # log_g
        (-1.0, 0.5),    # [Fe/H]
    ]
    for _ in sorted(GT_ABUNDANCES.keys()):
        bounds.append((-0.5, 0.5))  # [X/H]

    result = differential_evolution(
        objective, bounds, args=(wavelength, flux_obs),
        seed=123, maxiter=300, tol=1e-8, popsize=20,
        mutation=(0.5, 1.5), recombination=0.9,
        polish=True
    )
    T_eff, log_g, feh, abundances = unpack_params(result.x)
    flux_fit = synthesize_spectrum(wavelength, T_eff, log_g, feh, abundances)
    return T_eff, log_g, feh, abundances, flux_fit


# ====================================================================
# 4. Metrics
# ====================================================================
def compute_metrics(flux_gt, flux_fit, gt_params, fit_params):
    """PSNR, CC, and parameter relative errors."""
    # Spectrum metrics
    mse = np.mean((flux_gt - flux_fit)**2)
    psnr = 10.0 * np.log10(flux_gt.max()**2 / mse) if mse > 0 else 100.0
    cc = float(np.corrcoef(flux_gt, flux_fit)[0, 1])

    # Parameter relative errors
    param_names = ["T_eff", "log_g", "[Fe/H]"]
    gt_vals = [gt_params[0], gt_params[1], gt_params[2]]
    fit_vals = [fit_params[0], fit_params[1], fit_params[2]]
    
    param_errors = {}
    for name, gv, fv in zip(param_names, gt_vals, fit_vals):
        if abs(gv) > 1e-6:
            param_errors[f"RE_{name}"] = abs(fv - gv) / abs(gv)
        else:
            param_errors[f"RE_{name}"] = abs(fv - gv)

    # Abundance errors
    gt_ab = gt_params[3]
    fit_ab = fit_params[3]
    for elem in sorted(gt_ab.keys()):
        param_errors[f"AE_{elem}"] = abs(fit_ab[elem] - gt_ab[elem])

    return psnr, cc, param_errors


# ====================================================================
# 5. Visualization
# ====================================================================
def plot_results(wavelength, flux_gt, flux_obs, flux_fit,
                 gt_params, fit_params, param_errors):
    """4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: full spectrum overlay
    ax = axes[0, 0]
    ax.plot(wavelength, flux_obs, 'gray', alpha=0.4, lw=0.5, label='Observed')
    ax.plot(wavelength, flux_gt, 'b-', lw=1.0, label='GT spectrum')
    ax.plot(wavelength, flux_fit, 'r--', lw=1.0, label='Fitted spectrum')
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title("Stellar Spectrum: GT vs Fitted")
    ax.legend(fontsize=8)

    # Panel 2: zoom on Na D lines
    ax = axes[0, 1]
    mask = (wavelength > 5870) & (wavelength < 5920)
    ax.plot(wavelength[mask], flux_obs[mask], 'gray', alpha=0.5, lw=0.8, label='Observed')
    ax.plot(wavelength[mask], flux_gt[mask], 'b-', lw=1.2, label='GT')
    ax.plot(wavelength[mask], flux_fit[mask], 'r--', lw=1.2, label='Fitted')
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title("Zoom: Na D doublet (5890/5896 Å)")
    ax.legend(fontsize=8)

    # Panel 3: residuals
    ax = axes[1, 0]
    residual = flux_gt - flux_fit
    ax.plot(wavelength, residual, 'k-', lw=0.5)
    ax.axhline(0, color='r', ls='--', lw=0.5)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Residual (GT - Fit)")
    ax.set_title(f"Residuals | PSNR={param_errors['psnr']:.1f} dB, CC={param_errors['cc']:.4f}")

    # Panel 4: parameter comparison bar chart
    ax = axes[1, 1]
    labels = ["T_eff/1000", "log_g", "[Fe/H]+1"]
    gt_v = [gt_params[0]/1000, gt_params[1], gt_params[2]+1]
    fit_v = [fit_params[0]/1000, fit_params[1], fit_params[2]+1]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, gt_v, 0.3, label='GT', color='steelblue')
    ax.bar(x + 0.15, fit_v, 0.3, label='Fitted', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Stellar Parameters")
    ax.legend()

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
    print("Task 100: Stellar Spectral Synthesis (sme_stellar)")
    print("=" * 60)

    t0 = time.time()

    # Generate data
    print("\n[1] Generating ground-truth spectrum ...")
    flux_gt, flux_obs = generate_gt_data()

    # Inverse solve
    print("[2] Fitting stellar parameters (differential evolution) ...")
    T_fit, logg_fit, feh_fit, ab_fit, flux_fit = solve_inverse(WAVELENGTH, flux_obs)

    elapsed = time.time() - t0
    print(f"    Elapsed: {elapsed:.1f} s")

    # Parameters
    gt_params = (GT_TEFF, GT_LOGG, GT_FEH, GT_ABUNDANCES)
    fit_params = (T_fit, logg_fit, feh_fit, ab_fit)

    # Metrics
    print("[3] Computing metrics ...")
    psnr, cc, p_errors = compute_metrics(flux_gt, flux_fit, gt_params, fit_params)

    print(f"    Spectrum PSNR = {psnr:.2f} dB")
    print(f"    Spectrum CC   = {cc:.6f}")
    print(f"    T_eff: GT={GT_TEFF:.0f} K, Fit={T_fit:.0f} K, RE={p_errors['RE_T_eff']:.4f}")
    print(f"    log_g: GT={GT_LOGG:.2f}, Fit={logg_fit:.2f}, RE={p_errors['RE_log_g']:.4f}")
    print(f"    [Fe/H]: GT={GT_FEH:.2f}, Fit={feh_fit:.2f}")
    for elem in sorted(GT_ABUNDANCES.keys()):
        print(f"    [{elem}/H]: GT={GT_ABUNDANCES[elem]:.2f}, Fit={ab_fit[elem]:.2f}, AE={p_errors[f'AE_{elem}']:.4f}")

    # Build metrics dict
    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RE_Teff": float(p_errors["RE_T_eff"]),
        "RE_logg": float(p_errors["RE_log_g"]),
    }
    for elem in sorted(GT_ABUNDANCES.keys()):
        metrics[f"AE_{elem}"] = float(p_errors[f"AE_{elem}"])

    # Save
    print("[4] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), flux_gt)
        np.save(os.path.join(d, "recon_output.npy"), flux_fit)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Visualization
    print("[5] Plotting ...")
    extra_info = dict(p_errors)
    extra_info["psnr"] = psnr
    extra_info["cc"] = cc
    plot_results(WAVELENGTH, flux_gt, flux_obs, flux_fit,
                 gt_params, fit_params, extra_info)

    print(f"\n{'='*60}")
    print("Task 100 COMPLETE")
    print(f"{'='*60}")
    return metrics


if __name__ == "__main__":
    metrics = main()
