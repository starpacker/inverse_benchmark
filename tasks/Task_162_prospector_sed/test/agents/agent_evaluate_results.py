import os

import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

C_LIGHT = 2.998e18

H_PLANCK = 6.626e-27

K_BOLTZ = 1.381e-16

L_SUN = 3.828e33

PC_CM = 3.086e18

DIST_CM = 10.0 * PC_CM

FILTER_WAVES = np.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0,
                          12350.0, 16620.0, 21590.0])

FILTER_WIDTHS = np.array([560.0, 1380.0, 1370.0, 1530.0, 950.0,
                           1620.0, 2510.0, 2620.0])

RV_CALZETTI = 4.05

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

def forward_operator(params, filter_waves=None, filter_widths=None):
    """
    Compute model photometry in 8 bands given parameter vector.
    
    This is the forward model that maps physical parameters to observable fluxes.
    
    Parameters
    ----------
    params : array-like
        Parameter vector [log_mass, log_age, metallicity, Av]
    filter_waves : ndarray, optional
        Filter central wavelengths in Angstroms (default: FILTER_WAVES)
    filter_widths : ndarray, optional
        Filter bandwidths in Angstroms (default: FILTER_WIDTHS)
    
    Returns
    -------
    ndarray
        Array of flux densities (erg/s/cm^2/Hz) for each band
    """
    if filter_waves is None:
        filter_waves = FILTER_WAVES
    if filter_widths is None:
        filter_widths = FILTER_WIDTHS
    
    log_mass, log_age, metallicity, Av = params
    fluxes = np.zeros(len(filter_waves))
    
    for i in range(len(filter_waves)):
        # Integrate over filter bandwidth (trapezoidal, 50 points)
        w_lo = filter_waves[i] - filter_widths[i] / 2.0
        w_hi = filter_waves[i] + filter_widths[i] / 2.0
        wave_grid = np.linspace(w_lo, w_hi, 50)
        spec = composite_spectrum(wave_grid, log_mass, log_age, metallicity, Av)
        fluxes[i] = np.trapz(spec, wave_grid) / (w_hi - w_lo)
    
    return fluxes

def evaluate_results(data, inversion_result, results_dir=None):
    """
    Evaluate inversion results, compute metrics, save outputs, and create visualizations.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data
    inversion_result : dict
        Results dictionary from run_inversion
    results_dir : str, optional
        Directory to save results (default: ./results)
    
    Returns
    -------
    dict
        Metrics dictionary containing all evaluation metrics
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract data
    obs_flux = data['obs_flux']
    obs_unc = data['obs_unc']
    gt_flux = data['gt_flux']
    true_params = data['true_params']
    param_names = data['param_names']
    param_bounds = data['param_bounds']
    filter_names = data['filter_names']
    filter_waves = data['filter_waves']
    
    # Extract inversion results
    chain = inversion_result['chain']
    full_chain = inversion_result['full_chain']
    median_params = inversion_result['median_params']
    std_params = inversion_result['std_params']
    best_params = inversion_result['best_params']
    acc_frac = inversion_result['acceptance_fraction']
    elapsed = inversion_result['elapsed_time']
    nwalkers = inversion_result['nwalkers']
    nsteps = inversion_result['nsteps']
    nburn = inversion_result['nburn']
    
    ndim = len(param_names)
    
    # Compute reconstructed flux
    recon_flux = forward_operator(median_params)
    
    # Print parameter recovery
    print(f"\n[3] Posterior analysis ({chain.shape[0]} samples after burn-in)...")
    print(f"    {'Param':<14} {'True':>10} {'Median':>10} {'Std':>10} {'Best':>10}")
    print("    " + "-" * 54)
    for i, name in enumerate(param_names):
        print(f"    {name:<14} {true_params[i]:10.4f} {median_params[i]:10.4f} "
              f"{std_params[i]:10.4f} {best_params[i]:10.4f}")
    
    # Compute metrics
    print("\n[4] Computing metrics...")
    
    # Relative error for each parameter
    re_params = {}
    for i, name in enumerate(param_names):
        if abs(true_params[i]) > 1e-10:
            re = abs(median_params[i] - true_params[i]) / abs(true_params[i])
        else:
            re = abs(median_params[i] - true_params[i])
        re_params[name] = float(re)
        print(f"    RE({name}) = {re:.4f}")
    
    mean_re = np.mean(list(re_params.values()))
    print(f"    Mean RE = {mean_re:.4f}")
    
    # Flux cross-correlation
    cc = np.corrcoef(gt_flux, recon_flux)[0, 1]
    print(f"    Flux CC = {cc:.6f}")
    
    # Chi-squared
    chi2 = np.sum(((obs_flux - recon_flux) / obs_unc) ** 2)
    chi2_red = chi2 / (len(obs_flux) - ndim)
    print(f"    Reduced chi² = {chi2_red:.4f}")
    
    # Acceptance fraction
    print(f"    Acceptance fraction = {acc_frac:.3f}")
    
    # Build metrics dictionary
    metrics = {
        "task": "prospector_sed",
        "method": "analytic_SED_emcee_MCMC",
        "true_params": dict(zip(param_names, true_params.tolist())),
        "median_params": dict(zip(param_names, median_params.tolist())),
        "best_params": dict(zip(param_names, best_params.tolist())),
        "std_params": dict(zip(param_names, std_params.tolist())),
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
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n    Saved metrics → {metrics_path}")
    
    # Save data arrays
    print("\n[5] Saving data arrays...")
    
    # gt_output.npy: ground truth flux + true params
    gt_output = np.concatenate([gt_flux, true_params])
    np.save(os.path.join(results_dir, "gt_output.npy"), gt_output)
    
    # recon_output.npy: reconstructed flux + median params
    recon_output = np.concatenate([recon_flux, median_params])
    np.save(os.path.join(results_dir, "recon_output.npy"), recon_output)
    
    # Full posterior chain
    np.save(os.path.join(results_dir, "posterior_chain.npy"), chain)
    
    print(f"    gt_output.npy shape: {gt_output.shape}")
    print(f"    recon_output.npy shape: {recon_output.shape}")
    print(f"    posterior_chain.npy shape: {chain.shape}")
    
    # Create visualizations
    print("\n[6] Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # --- Panel 1: SED comparison ---
    ax1 = fig.add_subplot(2, 3, 1)
    wave_um = filter_waves / 1e4  # to microns
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
    for i, name in enumerate(filter_names):
        ax1.annotate(name, (wave_um[i], gt_flux[i]), fontsize=7,
                     textcoords="offset points", xytext=(0, 10), ha="center")
    
    # --- Panel 2: Residuals ---
    ax2 = fig.add_subplot(2, 3, 2)
    residual_sigma = (obs_flux - recon_flux) / obs_unc
    colors_resid = ["green" if abs(r) < 1 else "orange" if abs(r) < 2 else "red"
                    for r in residual_sigma]
    ax2.bar(filter_names, residual_sigma, color=colors_resid, edgecolor="k", alpha=0.8)
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
    normalised_true = (true_params - param_bounds[:, 0]) / (param_bounds[:, 1] - param_bounds[:, 0])
    normalised_med = (median_params - param_bounds[:, 0]) / (param_bounds[:, 1] - param_bounds[:, 0])
    normalised_std = std_params / (param_bounds[:, 1] - param_bounds[:, 0])
    
    ax3.errorbar(x_pos - 0.1, normalised_true, fmt="bs", ms=10, label="True")
    ax3.errorbar(x_pos + 0.1, normalised_med, yerr=normalised_std, fmt="r^",
                 ms=10, capsize=4, label="Recovered")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(param_names, fontsize=9)
    ax3.set_ylabel("Normalised Value", fontsize=11)
    ax3.set_title("Parameter Recovery", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Panel 4-6: Corner plot (2D marginals for key pairs) ---
    pairs = [(0, 1), (0, 3), (2, 3)]
    for pidx, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, 4 + pidx)
        ax.scatter(chain[::5, i], chain[::5, j], c="steelblue", s=1, alpha=0.2)
        ax.axvline(true_params[i], color="red", ls="--", lw=1.2)
        ax.axhline(true_params[j], color="red", ls="--", lw=1.2)
        ax.plot(true_params[i], true_params[j], "r*", ms=14, zorder=5)
        ax.plot(median_params[i], median_params[j], "g^", ms=10, zorder=5)
        ax.set_xlabel(param_names[i], fontsize=10)
        ax.set_ylabel(param_names[j], fontsize=10)
        ax.set_title(f"{param_names[i]} vs {param_names[j]}", fontsize=11,
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
    for i in range(ndim):
        ax = axes2[i]
        for w in range(nwalkers):
            ax.plot(full_chain[:, w, i], alpha=0.2, lw=0.5)
        ax.axhline(true_params[i], color="red", ls="--", lw=1.5)
        ax.axvline(nburn, color="black", ls=":", lw=1)
        ax.set_ylabel(param_names[i], fontsize=10)
    axes2[-1].set_xlabel("Step", fontsize=11)
    fig2.suptitle("MCMC Chain Traces", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "chain_traces.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"    Saved chain traces → {os.path.join(results_dir, 'chain_traces.png')}")
    
    # Print summary
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
