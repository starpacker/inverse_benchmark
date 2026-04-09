"""
POSEIDON-inspired — Exoplanet Atmospheric Retrieval Inverse Problem
====================================================================
Task: Recover atmospheric composition and temperature from a
      transmission spectrum of a transiting exoplanet.

Inverse Problem:
    Given an observed wavelength-dependent transit depth spectrum D(λ),
    retrieve the atmospheric mixing ratios of absorbing species
    (H₂O, CH₄, CO₂) and the isothermal temperature T, plus the
    reference planetary radius R_p.

Forward Model (Simplified Transmission Spectroscopy):
    1. Set up a 1D isothermal atmosphere with N_layers pressure layers
       from P_top to P_bottom (log-spaced).
    2. At each layer, compute number density via ideal gas law:
       n(P) = P / (k_B T)
    3. For each wavelength λ, compute absorption cross-section σ(λ)
       as a sum of Gaussian absorption bands for each species.
    4. Compute slant optical depth τ(λ, z) through the limb geometry.
    5. Integrate to get the effective planetary radius R_eff(λ).
    6. Transit depth D(λ) = (R_eff(λ) / R_star)²

Inverse Solver:
    scipy.optimize.differential_evolution (global) with
    L-BFGS-B local refinement.

Repo inspiration: https://github.com/MartianColonist/POSEIDON
Reference: MacDonald & Madhusudhan (2017), MNRAS.

Usage:
    /data/yjh/poseidon_atm_env/bin/python poseidon_atm_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import differential_evolution, minimize

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physical constants (SI)
K_B = 1.380649e-23       # Boltzmann constant [J/K]
AMU = 1.66054e-27        # atomic mass unit [kg]
R_SUN = 6.9634e8         # Solar radius [m]
R_JUP = 7.1492e7         # Jupiter radius [m]
G = 6.674e-11            # gravitational constant

# Stellar and planetary system parameters (hot Jupiter)
R_STAR = 1.0 * R_SUN     # stellar radius [m]
M_PLANET = 1.0 * 1.898e27  # planet mass ~ 1 M_Jup [kg]

# Atmosphere grid
N_LAYERS = 60            # number of atmospheric layers
P_TOP = 1e-4             # top pressure [Pa] ~ 1e-9 bar → use 1e-4 Pa
P_BOTTOM = 1e6           # bottom pressure [Pa] ~ 10 bar
SCALE_HEIGHT_REF = 500e3  # reference scale height ~500 km (updated in forward)

# Wavelength grid (near-infrared, 0.8 – 5.0 μm)
N_WAVE = 200
WAV_MIN = 0.8e-6         # [m]
WAV_MAX = 5.0e-6         # [m]

# Noise
NOISE_LEVEL = 30e-6      # transit depth noise (30 ppm)
SEED = 42

# Ground truth atmospheric parameters
GT_PARAMS = {
    "T":          1200.0,     # isothermal temperature [K]
    "log_X_H2O":  -3.5,       # log10 mixing ratio of H₂O
    "log_X_CH4":  -4.0,       # log10 mixing ratio of CH₄
    "log_X_CO2":  -5.0,       # log10 mixing ratio of CO₂
    "R_p":        1.2 * R_JUP # reference planet radius [m]
}

# Mean molecular weight of H₂/He atmosphere
MU_ATM = 2.3 * AMU  # mean molecular mass [kg]

# Species absorption band definitions
# Each species: list of (center_wavelength [m], width [m], peak_cross_section [m²])
SPECIES_BANDS = {
    "H2O": [
        (1.4e-6,  0.15e-6,  1.0e-25),
        (1.85e-6, 0.12e-6,  6.0e-26),
        (2.7e-6,  0.20e-6,  1.5e-25),
    ],
    "CH4": [
        (1.65e-6, 0.10e-6,  8.0e-26),
        (2.3e-6,  0.15e-6,  5.0e-26),
        (3.3e-6,  0.25e-6,  1.2e-25),
    ],
    "CO2": [
        (4.3e-6,  0.20e-6,  2.0e-25),
        (2.0e-6,  0.08e-6,  2.0e-26),
    ],
}

# Rayleigh scattering cross-section reference (H₂)
SIGMA_RAY_REF = 5.31e-31  # at 350 nm reference
WAV_RAY_REF = 0.35e-6


# ═══════════════════════════════════════════════════════════
# 2. Forward Operator (Transmission Spectroscopy)
# ═══════════════════════════════════════════════════════════
def compute_cross_section(wavelengths, species_name):
    """
    Compute simplified absorption cross-section for a species.

    Uses sum of Gaussian absorption bands. Real opacity databases
    (e.g., ExoMol, HITRAN) have millions of lines; this simplified
    model captures the essential wavelength-dependent structure.

    Parameters
    ----------
    wavelengths : np.ndarray  Wavelength array [m].
    species_name : str  Species name ('H2O', 'CH4', 'CO2').

    Returns
    -------
    sigma : np.ndarray  Cross-section array [m²] of shape (N_wave,).
    """
    bands = SPECIES_BANDS[species_name]
    sigma = np.zeros_like(wavelengths)
    for center, width, peak in bands:
        sigma += peak * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return sigma


def compute_rayleigh(wavelengths):
    """
    Rayleigh scattering cross-section for H₂ (λ⁻⁴ dependence).
    """
    return SIGMA_RAY_REF * (WAV_RAY_REF / wavelengths) ** 4


def forward_operator(params, wavelengths):
    """
    Compute the transmission spectrum (transit depth vs wavelength).

    Implements a simplified version of the atmospheric transmission
    calculation as in POSEIDON/Exo-Transmit/petitRADTRANS:

    1. Build pressure-altitude grid assuming hydrostatic equilibrium.
    2. At each layer, compute number densities of absorbers.
    3. For each wavelength, compute slant optical depth through each
       annular ring of atmosphere.
    4. Integrate to get the effective transit radius R_eff(λ).
    5. Transit depth D(λ) = (R_eff(λ)/R_star)².

    Parameters
    ----------
    params : dict  Atmospheric parameters.
    wavelengths : np.ndarray  Wavelength array [m].

    Returns
    -------
    transit_depth : np.ndarray  Transit depth D(λ).
    """
    T = params["T"]
    X_H2O = 10.0 ** params["log_X_H2O"]
    X_CH4 = 10.0 ** params["log_X_CH4"]
    X_CO2 = 10.0 ** params["log_X_CO2"]
    R_p = params["R_p"]

    # Pressure grid (log-spaced, top to bottom)
    pressures = np.logspace(np.log10(P_TOP), np.log10(P_BOTTOM), N_LAYERS)

    # Scale height H = k_B T / (mu g)
    g = G * M_PLANET / R_p ** 2  # surface gravity
    H = K_B * T / (MU_ATM * g)

    # Altitude grid from hydrostatic equilibrium: z = -H ln(P/P_ref)
    P_ref = pressures[-1]  # reference at bottom
    altitudes = -H * np.log(pressures / P_ref)  # z=0 at bottom

    # Layer boundaries (midpoints between levels)
    alt_boundaries = np.zeros(N_LAYERS + 1)
    alt_boundaries[0] = altitudes[0] + 0.5 * (altitudes[0] - altitudes[1])
    alt_boundaries[-1] = altitudes[-1] - 0.5 * (altitudes[-2] - altitudes[-1])
    alt_boundaries[1:-1] = 0.5 * (altitudes[:-1] + altitudes[1:])
    dz = np.abs(np.diff(alt_boundaries))  # layer thicknesses

    # Number densities [m⁻³]
    n_total = pressures / (K_B * T)
    n_H2O = X_H2O * n_total
    n_CH4 = X_CH4 * n_total
    n_CO2 = X_CO2 * n_total

    # Cross-sections
    sigma_H2O = compute_cross_section(wavelengths, "H2O")
    sigma_CH4 = compute_cross_section(wavelengths, "CH4")
    sigma_CO2 = compute_cross_section(wavelengths, "CO2")
    sigma_ray = compute_rayleigh(wavelengths)

    # Effective radius calculation
    # For each layer j, the annulus at radius r_j = R_p + z_j
    # contributes an area of ~2π r_j dr_j × (1 - exp(-τ_j))
    # to the total blocking area
    r = R_p + altitudes  # radius of each layer center

    # Transit depth: D(λ) = [R_p² + 2∫ r(1-exp(-τ)) dr] / R_star²
    # We compute τ along slant path through each annulus
    transit_depth = np.zeros(len(wavelengths))

    for j in range(N_LAYERS):
        # Total extinction at layer j for each wavelength
        kappa = (n_H2O[j] * sigma_H2O +
                 n_CH4[j] * sigma_CH4 +
                 n_CO2[j] * sigma_CO2 +
                 n_total[j] * sigma_ray)  # shape (N_wave,)

        # Slant path length through annulus at impact parameter b = r[j]
        # ds ≈ 2 * sqrt(2 * r[j] * H) for isothermal atmosphere
        ds = 2.0 * np.sqrt(2.0 * r[j] * H)

        # Optical depth along slant
        tau = kappa * ds  # shape (N_wave,)

        # Contribution to effective area
        transit_depth += 2.0 * r[j] * dz[j] * (1.0 - np.exp(-tau))

    # Add opaque disk of planet
    transit_depth = (R_p ** 2 + transit_depth) / R_STAR ** 2

    return transit_depth


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def generate_data(wavelengths):
    """
    Generate synthetic observed transmission spectrum.

    Returns
    -------
    spectrum_obs : np.ndarray  Noisy transit depth.
    spectrum_clean : np.ndarray  Clean transit depth.
    sigma_obs : np.ndarray  Noise standard deviation per point.
    """
    rng = np.random.RandomState(SEED)
    spectrum_clean = forward_operator(GT_PARAMS, wavelengths)
    sigma_obs = NOISE_LEVEL * np.ones(len(wavelengths))
    noise = rng.normal(0, sigma_obs)
    spectrum_obs = spectrum_clean + noise
    return spectrum_obs, spectrum_clean, sigma_obs


def load_or_generate_data():
    """
    Load cached data or generate fresh.
    """
    wavelengths = np.linspace(WAV_MIN, WAV_MAX, N_WAVE)
    data_file = os.path.join(RESULTS_DIR, "observed_spectrum.npz")

    if os.path.exists(data_file):
        print("[DATA] Loading cached data ...")
        data = np.load(data_file)
        return (wavelengths, data["spectrum_obs"],
                data["spectrum_clean"], data["sigma_obs"])

    print("[DATA] Generating synthetic transmission spectrum ...")
    spectrum_obs, spectrum_clean, sigma_obs = generate_data(wavelengths)

    np.savez(data_file,
             wavelengths=wavelengths,
             spectrum_obs=spectrum_obs,
             spectrum_clean=spectrum_clean,
             sigma_obs=sigma_obs)
    print(f"[DATA] Saved → {data_file}")
    return wavelengths, spectrum_obs, spectrum_clean, sigma_obs


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver (Atmospheric Retrieval)
# ═══════════════════════════════════════════════════════════
def params_from_vector(x):
    """
    Convert parameter vector to dict.

    x = [T, log_X_H2O, log_X_CH4, log_X_CO2, R_p_factor]
    where R_p = R_p_factor × R_JUP
    """
    return {
        "T":         x[0],
        "log_X_H2O": x[1],
        "log_X_CH4": x[2],
        "log_X_CO2": x[3],
        "R_p":       x[4] * R_JUP,
    }


def vector_from_params(params):
    """
    Convert parameter dict to vector.
    """
    return np.array([
        params["T"],
        params["log_X_H2O"],
        params["log_X_CH4"],
        params["log_X_CO2"],
        params["R_p"] / R_JUP,
    ])


# Parameter bounds
PARAM_BOUNDS = [
    (500.0, 2500.0),     # T [K]
    (-8.0, -1.0),        # log_X_H2O
    (-8.0, -1.0),        # log_X_CH4
    (-8.0, -1.0),        # log_X_CO2
    (0.8, 2.0),          # R_p / R_JUP
]

PARAM_NAMES = ["T", "log_X_H2O", "log_X_CH4", "log_X_CO2", "R_p/R_Jup"]


def chi_squared(x, wavelengths, spectrum_obs, sigma_obs):
    """
    Chi-squared cost function for atmospheric retrieval.
    """
    params = params_from_vector(x)
    model = forward_operator(params, wavelengths)
    return np.sum(((spectrum_obs - model) / sigma_obs) ** 2)


def reconstruct(wavelengths, spectrum_obs, sigma_obs):
    """
    Perform atmospheric retrieval via global + local optimization.

    Phase 1: Differential evolution for global search.
    Phase 2: L-BFGS-B refinement from DE solution.

    Returns
    -------
    fit_params : dict  Retrieved atmospheric parameters.
    spectrum_fit : np.ndarray  Model spectrum from retrieved parameters.
    """
    print("[RECON] Phase 1: Differential evolution (global search) ...")
    result_de = differential_evolution(
        chi_squared,
        bounds=PARAM_BOUNDS,
        args=(wavelengths, spectrum_obs, sigma_obs),
        seed=SEED,
        maxiter=300,
        tol=1e-8,
        polish=False,
        popsize=20,
    )
    print(f"  DE result: χ² = {result_de.fun:.2f}, success = {result_de.success}")

    print("[RECON] Phase 2: L-BFGS-B local refinement ...")
    result_local = minimize(
        chi_squared,
        x0=result_de.x,
        args=(wavelengths, spectrum_obs, sigma_obs),
        method="L-BFGS-B",
        bounds=PARAM_BOUNDS,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    print(f"  L-BFGS-B result: χ² = {result_local.fun:.2f}, success = {result_local.success}")

    fit_params = params_from_vector(result_local.x)
    spectrum_fit = forward_operator(fit_params, wavelengths)

    print("\n  Retrieved parameters:")
    gt_vec = vector_from_params(GT_PARAMS)
    fit_vec = result_local.x
    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:12s}: GT = {gt_vec[i]:10.4f}, Fit = {fit_vec[i]:10.4f}")

    return fit_params, spectrum_fit


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_params, fit_params, spectrum_clean, spectrum_fit):
    """
    Compute spectrum-fit and parameter-recovery metrics.

    Metrics:
      - PSNR: Peak signal-to-noise ratio of spectrum fit
      - CC: Pearson correlation coefficient
      - RMSE: Root mean square error of spectrum
      - RE: Relative error (norm)
      - Parameter-level relative errors
    """
    from skimage.metrics import structural_similarity as ssim_fn

    residual = spectrum_clean - spectrum_fit
    mse = np.mean(residual ** 2)
    rmse = float(np.sqrt(mse))

    # CC
    cc = float(np.corrcoef(spectrum_clean, spectrum_fit)[0, 1])

    # PSNR
    data_range = spectrum_clean.max() - spectrum_clean.min()
    psnr = float(10.0 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # SSIM (tile 1D to 2D for skimage)
    tile_rows = 7
    a2d = np.tile(spectrum_clean, (tile_rows, 1))
    b2d = np.tile(spectrum_fit, (tile_rows, 1))
    ssim = float(ssim_fn(a2d, b2d, data_range=data_range, win_size=7))

    # Relative error
    norm_gt = np.linalg.norm(spectrum_clean)
    re = float(np.linalg.norm(residual) / max(norm_gt, 1e-12))

    # Parameter recovery metrics
    param_keys = ["T", "log_X_H2O", "log_X_CH4", "log_X_CO2"]
    param_metrics = {}
    for k in param_keys:
        gt_v = gt_params[k]
        fit_v = fit_params[k]
        param_metrics[f"gt_{k}"] = float(gt_v)
        param_metrics[f"fit_{k}"] = float(fit_v)
        param_metrics[f"abs_err_{k}"] = float(abs(gt_v - fit_v))
        if abs(gt_v) > 1e-12:
            param_metrics[f"rel_err_{k}_pct"] = float(
                abs(gt_v - fit_v) / abs(gt_v) * 100
            )

    # Also compare R_p
    gt_rp = gt_params["R_p"] / R_JUP
    fit_rp = fit_params["R_p"] / R_JUP
    param_metrics["gt_R_p_Rjup"] = float(gt_rp)
    param_metrics["fit_R_p_Rjup"] = float(fit_rp)
    param_metrics["rel_err_R_p_pct"] = float(abs(gt_rp - fit_rp) / gt_rp * 100)

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
def visualize_results(wavelengths, spectrum_obs, spectrum_clean, spectrum_fit,
                      gt_params, fit_params, metrics, save_path):
    """
    Generate a 4-panel visualization of the retrieval results.
    """
    wav_um = wavelengths * 1e6  # convert to μm
    depth_ppm_obs = spectrum_obs * 1e6
    depth_ppm_clean = spectrum_clean * 1e6
    depth_ppm_fit = spectrum_fit * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Transmission spectrum
    ax = axes[0, 0]
    ax.scatter(wav_um, depth_ppm_obs, s=3, c='gray', alpha=0.4, label='Noisy obs')
    ax.plot(wav_um, depth_ppm_clean, 'b-', lw=1.5, label='Ground truth')
    ax.plot(wav_um, depth_ppm_fit, 'r--', lw=1.5, label='Retrieved')
    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel('Transit Depth [ppm]')
    ax.set_title('(a) Transmission Spectrum')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Residuals
    ax = axes[0, 1]
    res_ppm = (depth_ppm_clean - depth_ppm_fit)
    ax.plot(wav_um, res_ppm, 'g-', lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel('Residual [ppm]')
    ax.set_title(f'(b) Residuals  RMSE = {metrics["RMSE"]*1e6:.2f} ppm')
    ax.grid(True, alpha=0.3)

    # (c) Absorption feature zoom (1.0–3.0 μm)
    ax = axes[1, 0]
    mask = (wav_um >= 1.0) & (wav_um <= 3.5)
    ax.plot(wav_um[mask], depth_ppm_clean[mask], 'b-', lw=2, label='GT')
    ax.plot(wav_um[mask], depth_ppm_fit[mask], 'r--', lw=2, label='Retrieved')

    # Annotate absorption bands
    band_labels = [
        (1.4, 'H₂O'), (1.65, 'CH₄'), (1.85, 'H₂O'),
        (2.3, 'CH₄'), (2.7, 'H₂O'),
    ]
    ymin, ymax = ax.get_ylim()
    for bwav, blabel in band_labels:
        if 1.0 <= bwav <= 3.5:
            ax.axvline(bwav, color='purple', alpha=0.3, ls=':')
            ax.text(bwav, ymax - 0.05 * (ymax - ymin), blabel,
                    ha='center', va='top', fontsize=7, color='purple')
    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel('Transit Depth [ppm]')
    ax.set_title('(c) Absorption Features (1–3.5 μm)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Retrieved vs GT parameters (bar chart)
    ax = axes[1, 1]
    keys = ["T", "log_X_H2O", "log_X_CH4", "log_X_CO2", "R_p/R_Jup"]
    gt_vals = [
        gt_params["T"],
        gt_params["log_X_H2O"],
        gt_params["log_X_CH4"],
        gt_params["log_X_CO2"],
        gt_params["R_p"] / R_JUP,
    ]
    fit_vals = [
        fit_params["T"],
        fit_params["log_X_H2O"],
        fit_params["log_X_CH4"],
        fit_params["log_X_CO2"],
        fit_params["R_p"] / R_JUP,
    ]
    # Normalize for visualization (different scales)
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, gt_vals, w, label='GT', color='steelblue')
    ax.bar(x + w / 2, fit_vals, w, label='Retrieved', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=8, rotation=15)
    ax.set_title('(d) Parameter Recovery')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"POSEIDON-inspired — Exoplanet Atmospheric Retrieval\n"
        f"PSNR = {metrics['PSNR']:.1f} dB  |  "
        f"SSIM = {metrics['SSIM']:.4f}  |  "
        f"CC = {metrics['CC']:.6f}  |  "
        f"RE = {metrics['RE']:.2e}",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  POSEIDON-inspired — Exoplanet Atmospheric Retrieval")
    print("=" * 65)

    wavelengths, spectrum_obs, spectrum_clean, sigma_obs = load_or_generate_data()

    print(f"\n[INFO] Wavelength range: {WAV_MIN*1e6:.1f} – {WAV_MAX*1e6:.1f} μm")
    print(f"[INFO] Number of spectral points: {N_WAVE}")
    print(f"[INFO] Noise level: {NOISE_LEVEL*1e6:.0f} ppm")

    print("\n[RECON] Starting atmospheric retrieval ...")
    fit_params, spectrum_fit = reconstruct(wavelengths, spectrum_obs, sigma_obs)

    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(GT_PARAMS, fit_params, spectrum_clean, spectrum_fit)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.6g}")
        else:
            print(f"  {k:30s} = {v}")

    # Save outputs
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), spectrum_clean)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), spectrum_fit)
    np.save(os.path.join(RESULTS_DIR, "measurements.npy"), spectrum_obs)

    print(f"\n[SAVE] gt_output.npy      → {RESULTS_DIR}")
    print(f"[SAVE] recon_output.npy   → {RESULTS_DIR}")
    print(f"[SAVE] measurements.npy   → {RESULTS_DIR}")

    visualize_results(
        wavelengths, spectrum_obs, spectrum_clean, spectrum_fit,
        GT_PARAMS, fit_params, metrics,
        os.path.join(RESULTS_DIR, "reconstruction_result.png"),
    )

    print("\n" + "=" * 65)
    print("  DONE — Atmospheric retrieval benchmark complete")
    print("=" * 65)
