"""
radvel — Radial Velocity Keplerian Orbit Fitting
=================================================
Task: Recover Keplerian orbital parameters from radial velocity time series
Repo: https://github.com/California-Planet-Search/radvel
Paper: Fulton et al., "RadVel: The Radial Velocity Fitting Toolkit" (PASP, 2018)

Inverse Problem:
    Forward: Given Keplerian orbital elements (P, tp, e, ω, K), compute
             radial velocity time series v(t) = K[cos(ν(t)+ω) + e·cos(ω)] + γ
    Inverse: From noisy RV measurements v_obs(t), fit Keplerian orbital
             elements via maximum-likelihood optimization

Usage:
    /data/yjh/radvel_env/bin/python radvel_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# True orbital parameters for a 2-planet system
# Planet b: Hot Jupiter
TRUE_PARAMS = {
    'per1': 15.234,        # Period (days)
    'tp1': 2458200.5,      # Time of periastron (JD)
    'e1': 0.15,            # Eccentricity
    'w1': np.deg2rad(65),  # Argument of periastron (radians)
    'k1': 45.0,            # RV semi-amplitude (m/s)
    # Planet c: outer companion
    'per2': 105.7,         # Period (days)
    'tp2': 2458230.0,      # Time of periastron (JD)
    'e2': 0.30,            # Eccentricity
    'w2': np.deg2rad(210), # Argument of periastron (radians)
    'k2': 22.0,            # RV semi-amplitude (m/s)
    # Instrument parameters
    'gamma': -3.5,         # Systemic velocity (m/s)
    'jit': 2.0,            # Jitter (m/s)
}

# Observation parameters
N_OBS = 80                 # Number of observations
T_SPAN = 300.0             # Observation time span (days)
T_START = 2458150.0        # Start time (JD)
RV_ERR = 3.0               # Measurement uncertainty (m/s)


# ═══════════════════════════════════════════════════════════
# 2. Forward Model & Synthetic Data
# ═══════════════════════════════════════════════════════════
def forward_model(t, params_dict):
    """
    Forward: Compute RV time series from orbital parameters.
    
    For each planet: v_i(t) = K_i [cos(ν_i(t) + ω_i) + e_i cos(ω_i)]
    Total: v(t) = Σ v_i(t) + γ
    """
    import radvel
    
    rv_total = np.zeros_like(t)
    
    # Planet 1
    orbel1 = np.array([params_dict['per1'], params_dict['tp1'],
                        params_dict['e1'], params_dict['w1'],
                        params_dict['k1']])
    rv_total += radvel.kepler.rv_drive(t, orbel1)
    
    # Planet 2
    orbel2 = np.array([params_dict['per2'], params_dict['tp2'],
                        params_dict['e2'], params_dict['w2'],
                        params_dict['k2']])
    rv_total += radvel.kepler.rv_drive(t, orbel2)
    
    # Systemic velocity
    rv_total += params_dict['gamma']
    
    return rv_total


def generate_synthetic_data():
    """
    Generate synthetic radial velocity observations.
    Includes realistic irregular time sampling and Gaussian noise.
    """
    # Irregular time sampling (simulating real observing cadence)
    t = np.sort(T_START + T_SPAN * np.random.rand(N_OBS))
    
    # Add some observing gaps (simulating weather, telescope scheduling)
    # Remove points in certain windows
    mask = ~((t > T_START + 60) & (t < T_START + 75))  # 15-day gap
    mask &= ~((t > T_START + 180) & (t < T_START + 190))  # 10-day gap
    t = t[mask]
    
    # Compute true RV
    rv_true = forward_model(t, TRUE_PARAMS)
    
    # Add noise (measurement error + jitter)
    total_err = np.sqrt(RV_ERR**2 + TRUE_PARAMS['jit']**2)
    rv_err = np.ones(len(t)) * RV_ERR  # formal errors
    noise = total_err * np.random.randn(len(t))
    rv_obs = rv_true + noise
    
    print(f"  Generated {len(t)} observations over {t[-1]-t[0]:.1f} days")
    print(f"  RV range: [{rv_obs.min():.1f}, {rv_obs.max():.1f}] m/s")
    print(f"  True jitter: {TRUE_PARAMS['jit']:.1f} m/s, measurement σ: {RV_ERR:.1f} m/s")
    
    return t, rv_obs, rv_err, rv_true


# ═══════════════════════════════════════════════════════════
# 3. Inverse Solver: Maximum Likelihood Keplerian Fit
# ═══════════════════════════════════════════════════════════
def reconstruct(t, rv_obs, rv_err):
    """
    Inverse: Fit Keplerian orbital parameters from RV data.
    
    Uses radvel's Likelihood + Posterior framework with
    maximum-likelihood optimization (Powell method).
    """
    import radvel
    
    nplanets = 2
    
    # Set up Parameters in 'per tp e w k' basis
    params = radvel.Parameters(nplanets, basis='per tp e w k')
    
    # Initial guesses (slightly perturbed from truth to simulate realistic fitting)
    params['per1'] = radvel.Parameter(value=15.5)    # ~2% off
    params['tp1'] = radvel.Parameter(value=2458201.0)
    params['e1'] = radvel.Parameter(value=0.1)       # start lower
    params['w1'] = radvel.Parameter(value=np.deg2rad(70))  # ~5 deg off
    params['k1'] = radvel.Parameter(value=40.0)      # ~11% off
    
    params['per2'] = radvel.Parameter(value=110.0)   # ~4% off
    params['tp2'] = radvel.Parameter(value=2458225.0)
    params['e2'] = radvel.Parameter(value=0.25)      # start lower
    params['w2'] = radvel.Parameter(value=np.deg2rad(200))  # ~10 deg off
    params['k2'] = radvel.Parameter(value=18.0)      # ~18% off
    
    params['dvdt'] = radvel.Parameter(value=0.0)     # no linear trend
    params['curv'] = radvel.Parameter(value=0.0)     # no curvature
    
    # Create the RV model
    mod = radvel.RVModel(params, time_base=np.median(t))
    
    # Create likelihood
    like = radvel.likelihood.RVLikelihood(mod, t, rv_obs, rv_err)
    
    # Set gamma and jitter as free parameters
    like.params['gamma'] = radvel.Parameter(value=0.0)
    like.params['jit'] = radvel.Parameter(value=1.0)
    
    # Set up priors
    post = radvel.posterior.Posterior(like)
    
    # Add eccentricity prior (avoid unphysical e > 1)
    post.priors += [radvel.prior.EccentricityPrior(nplanets)]
    
    # Add positive K prior
    post.priors += [radvel.prior.PositiveKPrior(nplanets)]
    
    # Hard bounds on jitter
    post.priors += [radvel.prior.HardBounds('jit', 0.01, 20.0)]
    
    print("  [FIT] Running maximum-likelihood optimization...")
    print(f"  [FIT] Initial guess - Per1={params['per1'].value:.2f}, K1={params['k1'].value:.1f}")
    print(f"  [FIT] Initial guess - Per2={params['per2'].value:.2f}, K2={params['k2'].value:.1f}")
    
    # Fit
    res = radvel.fitting.maxlike_fitting(post, verbose=False, method='Powell')
    
    # Extract fitted parameters
    fitted = {}
    for key in res.params:
        fitted[key] = float(res.params[key].value)
    
    print(f"  [FIT] Fitted - Per1={fitted['per1']:.4f}, K1={fitted['k1']:.2f}")
    print(f"  [FIT] Fitted - Per2={fitted['per2']:.4f}, K2={fitted['k2']:.2f}")
    print(f"  [FIT] Fitted - gamma={fitted['gamma']:.2f}, jit={fitted['jit']:.2f}")
    
    return fitted, res


# ═══════════════════════════════════════════════════════════
# 4. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(t, rv_true, rv_obs, rv_err, fitted_params):
    """
    Evaluate the quality of Keplerian orbit fit.
    """
    # Compute fitted RV model
    rv_fitted = forward_model(t, fitted_params)
    
    # Residuals
    residuals = rv_obs - rv_fitted
    
    # RMS of residuals
    rms_residuals = np.sqrt(np.mean(residuals**2))
    
    # PSNR: compare fitted model to true RV
    mse_model = np.mean((rv_true - rv_fitted)**2)
    data_range = rv_true.max() - rv_true.min()
    psnr = 10 * np.log10(data_range**2 / mse_model) if mse_model > 0 else float('inf')
    
    # Correlation between true and fitted RV curves
    cc = np.corrcoef(rv_true, rv_fitted)[0, 1]
    
    # Parameter recovery errors
    param_errors = {}
    for key in ['per1', 'per2', 'k1', 'k2', 'e1', 'e2', 'gamma', 'jit']:
        true_val = TRUE_PARAMS[key]
        fit_val = fitted_params.get(key, 0)
        if key.startswith('w'):
            # Angular difference
            diff = np.abs(np.rad2deg(true_val) - np.rad2deg(fit_val))
            diff = min(diff, 360 - diff)
            param_errors[f'{key}_error_deg'] = float(diff)
        elif abs(true_val) > 1e-10:
            param_errors[f'{key}_rel_error'] = float(abs(fit_val - true_val) / abs(true_val))
        else:
            param_errors[f'{key}_abs_error'] = float(abs(fit_val - true_val))
    
    # Omega (in degrees)
    for i in [1, 2]:
        w_true = np.rad2deg(TRUE_PARAMS[f'w{i}'])
        w_fit = np.rad2deg(fitted_params.get(f'w{i}', 0))
        diff = abs(w_true - w_fit)
        diff = min(diff, 360 - diff)
        param_errors[f'w{i}_error_deg'] = float(diff)
    
    # Chi-squared
    total_err = np.sqrt(rv_err**2 + fitted_params.get('jit', 0)**2)
    chi2 = np.sum((residuals / total_err)**2)
    reduced_chi2 = chi2 / (len(t) - 12)  # approx DOF
    
    metrics = {
        'psnr': float(psnr),
        'cc': float(cc),
        'rms_residuals': float(rms_residuals),
        'reduced_chi2': float(reduced_chi2),
        **param_errors,
    }
    
    return metrics


# ═══════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(t, rv_obs, rv_err, rv_true, fitted_params, metrics, save_path):
    """Generate comprehensive visualization of RV orbit fitting."""
    rv_fitted = forward_model(t, fitted_params)
    residuals = rv_obs - rv_fitted
    
    # Fine time grid for smooth curves
    t_fine = np.linspace(t.min() - 5, t.max() + 5, 1000)
    rv_true_fine = forward_model(t_fine, TRUE_PARAMS)
    rv_fit_fine = forward_model(t_fine, fitted_params)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Full RV time series
    ax = axes[0, 0]
    ax.errorbar(t - T_START, rv_obs, yerr=rv_err, fmt='o', ms=4, 
                color='gray', alpha=0.6, label='Observed', zorder=1)
    ax.plot(t_fine - T_START, rv_true_fine, 'b-', lw=1, alpha=0.5, label='True')
    ax.plot(t_fine - T_START, rv_fit_fine, 'r-', lw=1.5, label='Fitted')
    ax.set_xlabel('Time (days from start)')
    ax.set_ylabel('RV (m/s)')
    ax.set_title('Radial Velocity Time Series')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (b) Phase-folded Planet 1
    ax = axes[0, 1]
    # Subtract planet 2 contribution
    rv_minus_p2 = rv_obs - forward_model(t, {**fitted_params, 'k1': 0, 'gamma': 0})
    phase1 = ((t - fitted_params['tp1']) % fitted_params['per1']) / fitted_params['per1']
    t_phase1 = np.linspace(0, 1, 200)
    t_phase1_full = fitted_params['tp1'] + t_phase1 * fitted_params['per1']
    rv_p1_model = forward_model(t_phase1_full, {**fitted_params, 'k2': 0, 'gamma': 0})
    
    ax.errorbar(phase1, rv_minus_p2, yerr=rv_err, fmt='o', ms=4, 
                color='steelblue', alpha=0.7)
    ax.plot(t_phase1, rv_p1_model, 'r-', lw=2)
    ax.set_xlabel('Phase')
    ax.set_ylabel('RV (m/s)')
    ax.set_title(f'Planet b (P={fitted_params["per1"]:.2f} d, K={fitted_params["k1"]:.1f} m/s)')
    ax.grid(True, alpha=0.3)
    
    # (c) Phase-folded Planet 2
    ax = axes[0, 2]
    rv_minus_p1 = rv_obs - forward_model(t, {**fitted_params, 'k2': 0, 'gamma': 0})
    phase2 = ((t - fitted_params['tp2']) % fitted_params['per2']) / fitted_params['per2']
    t_phase2 = np.linspace(0, 1, 200)
    t_phase2_full = fitted_params['tp2'] + t_phase2 * fitted_params['per2']
    rv_p2_model = forward_model(t_phase2_full, {**fitted_params, 'k1': 0, 'gamma': 0})
    
    ax.errorbar(phase2, rv_minus_p1, yerr=rv_err, fmt='o', ms=4,
                color='darkorange', alpha=0.7)
    ax.plot(t_phase2, rv_p2_model, 'r-', lw=2)
    ax.set_xlabel('Phase')
    ax.set_ylabel('RV (m/s)')
    ax.set_title(f'Planet c (P={fitted_params["per2"]:.2f} d, K={fitted_params["k2"]:.1f} m/s)')
    ax.grid(True, alpha=0.3)
    
    # (d) Residuals
    ax = axes[1, 0]
    ax.errorbar(t - T_START, residuals, yerr=rv_err, fmt='o', ms=4,
                color='gray', alpha=0.7)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('Time (days from start)')
    ax.set_ylabel('Residual (m/s)')
    ax.set_title(f'Residuals (RMS={metrics["rms_residuals"]:.2f} m/s)')
    ax.grid(True, alpha=0.3)
    
    # (e) True vs Fitted RV
    ax = axes[1, 1]
    ax.scatter(rv_true, rv_fitted, s=15, alpha=0.6, c='steelblue')
    lim = max(abs(rv_true).max(), abs(rv_fitted).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Identity')
    ax.set_xlabel('True RV (m/s)')
    ax.set_ylabel('Fitted RV (m/s)')
    ax.set_title(f'True vs Fitted (CC={metrics["cc"]:.6f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # (f) Parameter recovery bar chart
    ax = axes[1, 2]
    param_names = ['per1', 'per2', 'k1', 'k2', 'e1', 'e2']
    rel_errors = [metrics.get(f'{p}_rel_error', 0) * 100 for p in param_names]
    bars = ax.bar(range(len(param_names)), rel_errors, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(['P_b', 'P_c', 'K_b', 'K_c', 'e_b', 'e_c'], fontsize=9)
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Parameter Recovery Error')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, rel_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(
        f"radvel — Keplerian Orbit Fitting (2 planets)\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC={metrics['cc']:.6f} | "
        f"RMS_res={metrics['rms_residuals']:.2f} m/s | χ²_red={metrics['reduced_chi2']:.3f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 6. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  radvel — Keplerian RV Orbit Fitting")
    print("=" * 60)
    
    # (a) Generate synthetic RV data
    print("\n[DATA] Generating synthetic RV observations...")
    t, rv_obs, rv_err, rv_true = generate_synthetic_data()
    
    # (b) Run orbit fitting
    print("\n[RECON] Running Keplerian orbit fitting...")
    fitted_params, posterior = reconstruct(t, rv_obs, rv_err)
    
    # (c) Evaluate
    print("\n[EVAL] Computing evaluation metrics...")
    metrics = compute_metrics(t, rv_true, rv_obs, rv_err, fitted_params)
    
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMS residuals = {metrics['rms_residuals']:.4f} m/s")
    print(f"[EVAL] Reduced χ² = {metrics['reduced_chi2']:.4f}")
    print(f"[EVAL] Per1 rel error = {metrics['per1_rel_error']*100:.4f}%")
    print(f"[EVAL] Per2 rel error = {metrics['per2_rel_error']*100:.4f}%")
    print(f"[EVAL] K1 rel error = {metrics['k1_rel_error']*100:.4f}%")
    print(f"[EVAL] K2 rel error = {metrics['k2_rel_error']*100:.4f}%")
    print(f"[EVAL] e1 rel error = {metrics['e1_rel_error']*100:.4f}%")
    print(f"[EVAL] e2 rel error = {metrics['e2_rel_error']*100:.4f}%")
    print(f"[EVAL] ω1 error = {metrics['w1_error_deg']:.4f}°")
    print(f"[EVAL] ω2 error = {metrics['w2_error_deg']:.4f}°")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Save arrays
    np.save(os.path.join(RESULTS_DIR, "input.npy"),
            np.column_stack([t, rv_obs, rv_err]))
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), rv_true)
    rv_fitted = forward_model(t, fitted_params)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), rv_fitted)
    print(f"[SAVE] Input shape: ({len(t)}, 3) → input.npy")
    print(f"[SAVE] GT shape: ({len(rv_true)},) → ground_truth.npy")
    print(f"[SAVE] Recon shape: ({len(rv_fitted)},) → reconstruction.npy")
    
    # (f) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(t, rv_obs, rv_err, rv_true, fitted_params, metrics, vis_path)
    
    print("\n" + "=" * 60)
    print("  DONE — radvel Keplerian Orbit Fitting")
    print("=" * 60)
