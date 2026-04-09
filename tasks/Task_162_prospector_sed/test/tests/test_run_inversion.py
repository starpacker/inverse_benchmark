import sys
import os
import dill
import numpy as np
import traceback
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function from the agent module
from agent_run_inversion import run_inversion

# ============================================================================
# INJECTED REFEREE CODE (from Reference B - evaluate_results)
# ============================================================================

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
    """Calzetti et al. (2000) attenuation curve k'(lambda)."""
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
    """Simple mapping from age+Z to an effective temperature."""
    age_gyr = 10.0 ** (log_age - 9.0)
    T = 5500.0 * age_gyr ** (-0.18) * (metallicity / 0.02) ** 0.05
    return np.clip(T, 2500.0, 50000.0)

def composite_spectrum(wave_aa, log_mass, log_age, metallicity, Av):
    """Compute observed flux density F_nu (erg/s/cm^2/Hz) at 10 pc."""
    mass = 10.0 ** log_mass
    T_hot = effective_temperature(log_age, metallicity)
    T_cool = 0.55 * T_hot

    age_gyr = 10.0 ** (log_age - 9.0)
    f_hot = np.clip(0.8 - 0.15 * np.log10(age_gyr + 0.01), 0.2, 0.95)

    nu = C_LIGHT / wave_aa
    def planck_nu(T):
        x = H_PLANCK * nu / (K_BOLTZ * T)
        x = np.clip(x, 0, 500)
        return 2 * H_PLANCK * nu**3 / C_LIGHT**2 / (np.exp(x) - 1.0 + 1e-30)

    B_hot = planck_nu(T_hot)
    B_cool = planck_nu(T_cool)

    B_composite = f_hot * B_hot + (1.0 - f_hot) * B_cool

    sigma_SB = 5.670e-5
    L_bol_per_unit = sigma_SB * (f_hot * T_hot**4 + (1 - f_hot) * T_cool**4)
    L_target = mass * L_SUN / (4.0 * np.pi * DIST_CM**2)

    scale = L_target / (np.pi * L_bol_per_unit + 1e-30)
    F_nu = scale * B_composite

    atten = dust_attenuation(wave_aa, Av)
    F_nu *= atten

    return F_nu

def forward_operator(params, filter_waves=None, filter_widths=None):
    """Compute model photometry in 8 bands given parameter vector."""
    if filter_waves is None:
        filter_waves = FILTER_WAVES
    if filter_widths is None:
        filter_widths = FILTER_WIDTHS
    
    log_mass, log_age, metallicity, Av = params
    fluxes = np.zeros(len(filter_waves))
    
    for i in range(len(filter_waves)):
        w_lo = filter_waves[i] - filter_widths[i] / 2.0
        w_hi = filter_waves[i] + filter_widths[i] / 2.0
        wave_grid = np.linspace(w_lo, w_hi, 50)
        spec = composite_spectrum(wave_grid, log_mass, log_age, metallicity, Av)
        fluxes[i] = np.trapz(spec, wave_grid) / (w_hi - w_lo)
    
    return fluxes

def evaluate_results(data, inversion_result, results_dir=None):
    """
    Evaluate inversion results, compute metrics, save outputs, and create visualizations.
    
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
    
    gt_output = np.concatenate([gt_flux, true_params])
    np.save(os.path.join(results_dir, "gt_output.npy"), gt_output)
    
    recon_output = np.concatenate([recon_flux, median_params])
    np.save(os.path.join(results_dir, "recon_output.npy"), recon_output)
    
    np.save(os.path.join(results_dir, "posterior_chain.npy"), chain)
    
    print(f"    gt_output.npy shape: {gt_output.shape}")
    print(f"    recon_output.npy shape: {recon_output.shape}")
    print(f"    posterior_chain.npy shape: {chain.shape}")
    
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
    
    return metrics

# ============================================================================
# MAIN TEST LOGIC
# ============================================================================

def main():
    # Data paths provided
    data_paths = ['/data/yjh/prospector_sed_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 60)
    print("TEST: run_inversion Performance Validation")
    print("=" * 60)
    
    # Analyze data paths
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"\nOuter data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    print(f"\n[1] Loading outer data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"    Loaded successfully. Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"    args count: {len(args)}")
    print(f"    kwargs keys: {list(kwargs.keys())}")
    
    # Determine if this is direct execution or chained execution
    is_chained = len(inner_data_paths) > 0
    
    # Execute the agent function
    print(f"\n[2] Running agent run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
        print(f"    Agent run_inversion completed successfully.")
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine final result based on execution pattern
    if is_chained:
        # Chained execution: agent_output should be callable
        print(f"\n[2b] Chained execution detected. Processing inner data...")
        inner_data_path = inner_data_paths[0]
        
        with open(inner_data_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        if callable(agent_output):
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            final_result = agent_output
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    # The 'data' dictionary needed for evaluation should be the first argument
    # It contains obs_flux, obs_unc, true_params, etc.
    eval_data = args[0] if len(args) > 0 else kwargs.get('data', None)
    
    if eval_data is None:
        print("ERROR: Could not extract evaluation data from inputs!")
        sys.exit(1)
    
    print(f"\n[3] Evaluation data keys: {list(eval_data.keys())}")
    
    # Evaluate agent results
    print(f"\n[4] Evaluating AGENT results...")
    try:
        agent_metrics = evaluate_results(eval_data, final_result, results_dir="./results_agent")
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard results
    print(f"\n[5] Evaluating STANDARD results...")
    try:
        std_metrics = evaluate_results(eval_data, std_result, results_dir="./results_std")
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract key metrics for comparison
    # Primary metrics: flux_cross_correlation (higher is better), mean_relative_error (lower is better)
    agent_cc = agent_metrics['flux_cross_correlation']
    std_cc = std_metrics['flux_cross_correlation']
    
    agent_mre = agent_metrics['mean_relative_error']
    std_mre = std_metrics['mean_relative_error']
    
    agent_chi2 = agent_metrics['reduced_chi2']
    std_chi2 = std_metrics['reduced_chi2']
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Flux Cross-Correlation:")
    print(f"    Agent:    {agent_cc:.6f}")
    print(f"    Standard: {std_cc:.6f}")
    print(f"  Mean Relative Error:")
    print(f"    Agent:    {agent_mre:.4f}")
    print(f"    Standard: {std_mre:.4f}")
    print(f"  Reduced Chi²:")
    print(f"    Agent:    {agent_chi2:.4f}")
    print(f"    Standard: {std_chi2:.4f}")
    print("=" * 60)
    
    # Determine success criteria
    # Flux CC: Higher is better - agent should be at least 90% of standard
    # Mean RE: Lower is better - agent should not be more than 150% of standard
    # Chi2: Lower is better (closer to 1 is ideal) - agent should not be much worse
    
    passed = True
    tolerance_cc = 0.90  # Agent CC should be at least 90% of standard
    tolerance_mre = 1.50  # Agent MRE should not exceed 150% of standard
    tolerance_chi2 = 2.0  # Agent chi2 should not be more than 2x standard
    
    # Check flux cross-correlation
    if std_cc > 0:
        if agent_cc < std_cc * tolerance_cc:
            print(f"FAIL: Agent CC ({agent_cc:.6f}) < {tolerance_cc*100}% of Standard CC ({std_cc:.6f})")
            passed = False
        else:
            print(f"PASS: Flux Cross-Correlation within tolerance")
    
    # Check mean relative error
    if std_mre > 0:
        if agent_mre > std_mre * tolerance_mre:
            print(f"FAIL: Agent MRE ({agent_mre:.4f}) > {tolerance_mre*100}% of Standard MRE ({std_mre:.4f})")
            passed = False
        else:
            print(f"PASS: Mean Relative Error within tolerance")
    else:
        # If std_mre is very small, just check agent is reasonable
        if agent_mre > 0.5:  # 50% relative error is too high
            print(f"FAIL: Agent MRE ({agent_mre:.4f}) is too high")
            passed = False
        else:
            print(f"PASS: Mean Relative Error within tolerance")
    
    # Check reduced chi2 (should be close to 1, but allow some flexibility)
    if std_chi2 > 0:
        if agent_chi2 > std_chi2 * tolerance_chi2 and agent_chi2 > 5.0:
            print(f"WARNING: Agent chi2 ({agent_chi2:.4f}) is higher than expected")
            # Don't fail on chi2 alone if other metrics are good
        else:
            print(f"PASS: Reduced Chi² within tolerance")
    
    print("\n" + "=" * 60)
    if passed:
        print("OVERALL: TEST PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("OVERALL: TEST FAILED")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()