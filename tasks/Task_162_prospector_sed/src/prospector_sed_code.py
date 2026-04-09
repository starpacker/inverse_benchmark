#!/usr/bin/env python
"""
Task 162: prospector_sed — SED Fitting with Analytic Stellar Population Model + emcee MCMC

Uses a simplified analytic SED model (modified blackbody + Calzetti dust law)
to fit 8-band photometry (ugrizJHK) via emcee MCMC sampling.

Free parameters:
  - log_mass   : log10 of stellar mass in solar masses [8, 12]
  - log_age    : log10 of age in years [8, 10.2]
  - metallicity: Z (absolute metallicity) [0.001, 0.05]
  - Av         : dust attenuation in V-band magnitudes [0.0, 3.0]

No FSPS dependency — purely analytic model.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import emcee

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
C_LIGHT = 2.998e18       # Å/s
H_PLANCK = 6.626e-27     # erg·s
K_BOLTZ = 1.381e-16      # erg/K
L_SUN = 3.828e33         # erg/s
PC_CM = 3.086e18         # cm per parsec
DIST_CM = 10.0 * PC_CM   # 10 pc in cm (absolute magnitude distance)

# Filter effective wavelengths (Å) for ugrizJHK
FILTER_NAMES = ["u", "g", "r", "i", "z", "J", "H", "K"]
FILTER_WAVES = np.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0,
                          12350.0, 16620.0, 21590.0])
FILTER_WIDTHS = np.array([560.0, 1380.0, 1370.0, 1530.0, 950.0,
                           1620.0, 2510.0, 2620.0])

# Calzetti dust law reference wavelengths
RV_CALZETTI = 4.05

# ──────────────────────────────────────────────────────────────────────────────
# Analytic SED model
# ──────────────────────────────────────────────────────────────────────────────

def calzetti_kprime(wave_um):
    """Calzetti et al. (2000) attenuation curve k'(lambda).
    wave_um : wavelength in microns.
    Returns k'(lambda) such that A(lambda) = Av / Rv * k'(lambda).
    """
    k = np.zeros_like(wave_um)
    lo = (wave_um >= 0.12) & (wave_um < 0.63)
    hi = (wave_um >= 0.63) & (wave_um <= 2.20)

    k[lo] = (2.659 * (-2.156 + 1.509 / wave_um[lo]
              - 0.198 / wave_um[lo]**2
              + 0.011 / wave_um[lo]**3) + RV_CALZETTI)
    k[hi] = (2.659 * (-1.857 + 1.040 / wave_um[hi]) + RV_CALZETTI)
    k = np.clip(k, 0.0, None)
    return k


def dust_attenuation(wave_aa, Av):
    """Return flux attenuation factor (multiply flux by this) for given Av."""
    wave_um = wave_aa / 1e4
    kp = calzetti_kprime(wave_um)
    tau = Av * kp / RV_CALZETTI
    return 10.0 ** (-0.4 * tau)


def effective_temperature(log_age, metallicity):
    """Simple mapping from age+Z to an effective temperature for the
    composite stellar population.  This is a toy analytic approximation:
      T_eff ~ 5000 * (age/1e9)^{-0.15} * (Z/0.02)^{0.05}  K
    Young, metal-poor populations are hotter.
    """
    age_gyr = 10.0 ** (log_age - 9.0)  # age in Gyr
    T = 5500.0 * age_gyr ** (-0.18) * (metallicity / 0.02) ** 0.05
    return np.clip(T, 2500.0, 50000.0)


def composite_spectrum(wave_aa, log_mass, log_age, metallicity, Av):
    """Compute observed flux density F_nu (erg/s/cm^2/Hz) at 10 pc
    for the analytic SED model.

    The model is a two-component modified blackbody:
      - Hot component (young stars): T from age/Z mapping
      - Cool component: 0.6 * T  (older underlying population)
    Relative luminosity fraction: hot=0.7, cool=0.3 (varies with age).
    """
    mass = 10.0 ** log_mass  # solar masses
    T_hot = effective_temperature(log_age, metallicity)
    T_cool = 0.55 * T_hot

    # Younger populations have larger hot fraction
    age_gyr = 10.0 ** (log_age - 9.0)
    f_hot = np.clip(0.8 - 0.15 * np.log10(age_gyr + 0.01), 0.2, 0.95)

    # Planck function B_nu(T) in erg/s/cm^2/Hz/sr
    nu = C_LIGHT / wave_aa
    def planck_nu(T):
        x = H_PLANCK * nu / (K_BOLTZ * T)
        x = np.clip(x, 0, 500)  # prevent overflow
        return 2 * H_PLANCK * nu**3 / C_LIGHT**2 / (np.exp(x) - 1.0 + 1e-30)

    B_hot = planck_nu(T_hot)
    B_cool = planck_nu(T_cool)

    # Composite (arbitrary normalisation, will be scaled by mass)
    B_composite = f_hot * B_hot + (1.0 - f_hot) * B_cool

    # Mass-to-light scaling: L ~ mass * L_sun (simplified)
    # Normalise so that bolometric luminosity ~ mass * L_sun
    # Integrate B_composite over frequency to get sigma*T^4/pi type quantity
    # Then scale to match mass * L_sun at 10 pc
    sigma_SB = 5.670e-5  # erg/cm^2/s/K^4
    L_bol_per_unit = sigma_SB * (f_hot * T_hot**4 + (1 - f_hot) * T_cool**4)
    L_target = mass * L_SUN / (4.0 * np.pi * DIST_CM**2)

    # Scale spectrum
    scale = L_target / (np.pi * L_bol_per_unit + 1e-30)
    F_nu = scale * B_composite

    # Apply dust attenuation
    atten = dust_attenuation(wave_aa, Av)
    F_nu *= atten

    return F_nu


def model_photometry(params):
    """Compute model photometry in 8 bands given parameter vector.
    params = [log_mass, log_age, metallicity, Av]
    Returns array of 8 flux densities (erg/s/cm^2/Hz).
    """
    log_mass, log_age, metallicity, Av = params
    fluxes = np.zeros(len(FILTER_WAVES))
    for i in range(len(FILTER_WAVES)):
        # Integrate over filter bandwidth (trapezoidal, 50 points)
        w_lo = FILTER_WAVES[i] - FILTER_WIDTHS[i] / 2.0
        w_hi = FILTER_WAVES[i] + FILTER_WIDTHS[i] / 2.0
        wave_grid = np.linspace(w_lo, w_hi, 50)
        spec = composite_spectrum(wave_grid, log_mass, log_age, metallicity, Av)
        fluxes[i] = np.trapz(spec, wave_grid) / (w_hi - w_lo)
    return fluxes


# ──────────────────────────────────────────────────────────────────────────────
# MCMC setup
# ──────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["log_mass", "log_age", "metallicity", "Av"]
PARAM_BOUNDS = np.array([
    [8.0, 12.0],     # log_mass
    [8.0, 10.2],     # log_age (100 Myr – 16 Gyr)
    [0.001, 0.05],   # metallicity
    [0.0, 3.0],      # Av
])
TRUE_PARAMS = np.array([10.5, 9.5, 0.019, 0.8])  # ground truth


def log_prior(params):
    """Flat priors within bounds."""
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        if params[i] < lo or params[i] > hi:
            return -np.inf
    return 0.0


def log_likelihood(params, obs_flux, obs_unc):
    """Gaussian likelihood for photometry."""
    model = model_photometry(params)
    if np.any(model <= 0):
        return -np.inf
    chi2 = np.sum(((obs_flux - model) / obs_unc) ** 2)
    return -0.5 * chi2


def log_probability(params, obs_flux, obs_unc):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, obs_flux, obs_unc)
    return lp + ll


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Task 162: Prospector SED Fitting")
    print("=" * 60)

    # ── Synthesize ground-truth observations ──────────────────────────────
    print("\n[1] Generating synthetic photometry with known parameters...")
    print(f"    True params: {dict(zip(PARAM_NAMES, TRUE_PARAMS))}")

    gt_flux = model_photometry(TRUE_PARAMS)
    snr = 20.0  # S/N = 20 → 5% noise
    obs_unc = gt_flux / snr
    obs_flux = gt_flux + obs_unc * np.random.randn(len(gt_flux))

    print(f"    Bands: {FILTER_NAMES}")
    print(f"    GT flux (erg/s/cm²/Hz): {gt_flux}")
    print(f"    Obs flux: {obs_flux}")

    # ── MCMC ──────────────────────────────────────────────────────────────
    ndim = len(PARAM_NAMES)
    nwalkers = 24
    nsteps = 800
    nburn = 300

    print(f"\n[2] Running emcee MCMC: {nwalkers} walkers × {nsteps} steps (burn {nburn})...")

    # Initialize walkers near truth with small scatter
    p0 = np.array([TRUE_PARAMS + 0.05 * np.random.randn(ndim) *
                    (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])
                    for _ in range(nwalkers)])
    # Clip to bounds
    for i in range(ndim):
        p0[:, i] = np.clip(p0[:, i], PARAM_BOUNDS[i, 0] + 1e-6,
                            PARAM_BOUNDS[i, 1] - 1e-6)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                     args=(obs_flux, obs_unc))

    t0 = time.time()
    sampler.run_mcmc(p0, nsteps, progress=True)
    elapsed = time.time() - t0
    print(f"    MCMC completed in {elapsed:.1f}s")

    # ── Extract results ───────────────────────────────────────────────────
    chain = sampler.get_chain(discard=nburn, flat=True)
    print(f"\n[3] Posterior analysis ({chain.shape[0]} samples after burn-in)...")

    median_params = np.median(chain, axis=0)
    std_params = np.std(chain, axis=0)
    best_idx = np.argmax(sampler.get_log_prob(discard=nburn, flat=True))
    best_params = chain[best_idx]

    print(f"    {'Param':<14} {'True':>10} {'Median':>10} {'Std':>10} {'Best':>10}")
    print("    " + "-" * 54)
    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:<14} {TRUE_PARAMS[i]:10.4f} {median_params[i]:10.4f} "
              f"{std_params[i]:10.4f} {best_params[i]:10.4f}")

    # ── Compute metrics ───────────────────────────────────────────────────
    print("\n[4] Computing metrics...")

    # Relative error for each parameter
    re_params = {}
    for i, name in enumerate(PARAM_NAMES):
        if abs(TRUE_PARAMS[i]) > 1e-10:
            re = abs(median_params[i] - TRUE_PARAMS[i]) / abs(TRUE_PARAMS[i])
        else:
            re = abs(median_params[i] - TRUE_PARAMS[i])
        re_params[name] = float(re)
        print(f"    RE({name}) = {re:.4f}")

    mean_re = np.mean(list(re_params.values()))
    print(f"    Mean RE = {mean_re:.4f}")

    # Flux cross-correlation
    recon_flux = model_photometry(median_params)
    cc = np.corrcoef(gt_flux, recon_flux)[0, 1]
    print(f"    Flux CC = {cc:.6f}")

    # Chi-squared
    chi2 = np.sum(((obs_flux - recon_flux) / obs_unc) ** 2)
    chi2_red = chi2 / (len(obs_flux) - ndim)
    print(f"    Reduced chi² = {chi2_red:.4f}")

    # Acceptance fraction
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"    Acceptance fraction = {acc_frac:.3f}")

    metrics = {
        "task": "prospector_sed",
        "method": "analytic_SED_emcee_MCMC",
        "true_params": dict(zip(PARAM_NAMES, TRUE_PARAMS.tolist())),
        "median_params": dict(zip(PARAM_NAMES, median_params.tolist())),
        "best_params": dict(zip(PARAM_NAMES, best_params.tolist())),
        "std_params": dict(zip(PARAM_NAMES, std_params.tolist())),
        "relative_errors": re_params,
        "mean_relative_error": float(mean_re),
        "flux_cross_correlation": float(cc),
        "reduced_chi2": float(chi2_red),
        "acceptance_fraction": float(acc_frac),
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "nburn": nburn,
        "n_posterior_samples": int(chain.shape[0]),
        "mcmc_time_s": float(elapsed),
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n    Saved metrics → {metrics_path}")

    # ── Save data arrays ──────────────────────────────────────────────────
    print("\n[5] Saving data arrays...")

    # gt_output.npy: ground truth flux + true params
    gt_output = np.concatenate([gt_flux, TRUE_PARAMS])
    np.save(os.path.join(results_dir, "gt_output.npy"), gt_output)

    # recon_output.npy: reconstructed flux + median params
    recon_output = np.concatenate([recon_flux, median_params])
    np.save(os.path.join(results_dir, "recon_output.npy"), recon_output)

    # Full posterior chain
    np.save(os.path.join(results_dir, "posterior_chain.npy"), chain)

    print(f"    gt_output.npy shape: {gt_output.shape}")
    print(f"    recon_output.npy shape: {recon_output.shape}")
    print(f"    posterior_chain.npy shape: {chain.shape}")

    # ── Visualization ─────────────────────────────────────────────────────
    print("\n[6] Creating visualizations...")

    fig = plt.figure(figsize=(18, 14))

    # --- Panel 1: SED comparison ---
    ax1 = fig.add_subplot(2, 3, 1)
    wave_um = FILTER_WAVES / 1e4  # to microns
    ax1.errorbar(wave_um, obs_flux, yerr=obs_unc, fmt="ko", ms=8,
                 capsize=3, label="Observed", zorder=3)
    ax1.plot(wave_um, gt_flux, "bs--", ms=7, lw=1.5, label="Ground Truth", zorder=2)
    ax1.plot(wave_um, recon_flux, "r^-", ms=7, lw=1.5, label="Reconstructed", zorder=2)
    ax1.set_xlabel("Wavelength (μm)", fontsize=11)
    ax1.set_ylabel("F_ν (erg/s/cm²/Hz)", fontsize=11)
    ax1.set_title("SED Comparison", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    # Add band labels
    for i, name in enumerate(FILTER_NAMES):
        ax1.annotate(name, (wave_um[i], gt_flux[i]), fontsize=7,
                     textcoords="offset points", xytext=(0, 10), ha="center")

    # --- Panel 2: Residuals ---
    ax2 = fig.add_subplot(2, 3, 2)
    residual_sigma = (obs_flux - recon_flux) / obs_unc
    colors_resid = ["green" if abs(r) < 1 else "orange" if abs(r) < 2 else "red"
                    for r in residual_sigma]
    ax2.bar(FILTER_NAMES, residual_sigma, color=colors_resid, edgecolor="k", alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(1, color="gray", ls="--", lw=0.5)
    ax2.axhline(-1, color="gray", ls="--", lw=0.5)
    ax2.set_xlabel("Band", fontsize=11)
    ax2.set_ylabel("Residual (σ)", fontsize=11)
    ax2.set_title("Flux Residuals", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Parameter recovery ---
    ax3 = fig.add_subplot(2, 3, 3)
    x_pos = np.arange(ndim)
    normalised_true = (TRUE_PARAMS - PARAM_BOUNDS[:, 0]) / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])
    normalised_med = (median_params - PARAM_BOUNDS[:, 0]) / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])
    normalised_std = std_params / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])

    ax3.errorbar(x_pos - 0.1, normalised_true, fmt="bs", ms=10, label="True")
    ax3.errorbar(x_pos + 0.1, normalised_med, yerr=normalised_std, fmt="r^",
                 ms=10, capsize=4, label="Recovered")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(PARAM_NAMES, fontsize=9)
    ax3.set_ylabel("Normalised Value", fontsize=11)
    ax3.set_title("Parameter Recovery", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4-5: Corner plot (2D marginals for key pairs) ---
    pairs = [(0, 1), (0, 3), (2, 3)]
    for pidx, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, 4 + pidx)
        ax.scatter(chain[::5, i], chain[::5, j], c="steelblue", s=1, alpha=0.2)
        ax.axvline(TRUE_PARAMS[i], color="red", ls="--", lw=1.2)
        ax.axhline(TRUE_PARAMS[j], color="red", ls="--", lw=1.2)
        ax.plot(TRUE_PARAMS[i], TRUE_PARAMS[j], "r*", ms=14, zorder=5)
        ax.plot(median_params[i], median_params[j], "g^", ms=10, zorder=5)
        ax.set_xlabel(PARAM_NAMES[i], fontsize=10)
        ax.set_ylabel(PARAM_NAMES[j], fontsize=10)
        ax.set_title(f"{PARAM_NAMES[i]} vs {PARAM_NAMES[j]}", fontsize=11,
                     fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Task 162: Prospector SED Fitting — Analytic Model + emcee MCMC",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    vis_path = os.path.join(results_dir, "vis_result.png")
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved visualization → {vis_path}")

    # --- Chain trace plot ---
    fig2, axes2 = plt.subplots(ndim, 1, figsize=(12, 8), sharex=True)
    full_chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    for i in range(ndim):
        ax = axes2[i]
        for w in range(nwalkers):
            ax.plot(full_chain[:, w, i], alpha=0.2, lw=0.5)
        ax.axhline(TRUE_PARAMS[i], color="red", ls="--", lw=1.5)
        ax.axvline(nburn, color="black", ls=":", lw=1)
        ax.set_ylabel(PARAM_NAMES[i], fontsize=10)
    axes2[-1].set_xlabel("Step", fontsize=11)
    fig2.suptitle("MCMC Chain Traces", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "chain_traces.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"    Saved chain traces → {os.path.join(results_dir, 'chain_traces.png')}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Mean Relative Error : {mean_re:.4f}")
    print(f"  Flux CC             : {cc:.6f}")
    print(f"  Reduced χ²          : {chi2_red:.4f}")
    print(f"  Acceptance Fraction : {acc_frac:.3f}")
    print(f"  MCMC Time           : {elapsed:.1f}s")
    print("=" * 60)
    print("Task 162: prospector_sed COMPLETE")

    return metrics


if __name__ == "__main__":
    main()
