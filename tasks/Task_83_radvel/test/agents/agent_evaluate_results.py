import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

import warnings

warnings.filterwarnings('ignore')

import radvel

def forward_operator(t, params_dict):
    """
    Forward: Compute RV time series from orbital parameters.
    
    For each planet: v_i(t) = K_i [cos(ν_i(t) + ω_i) + e_i cos(ω_i)]
    Total: v(t) = Σ v_i(t) + γ
    
    Parameters
    ----------
    t : ndarray
        Time array (JD)
    params_dict : dict
        Dictionary containing orbital parameters:
        - per1, tp1, e1, w1, k1: Planet 1 parameters
        - per2, tp2, e2, w2, k2: Planet 2 parameters
        - gamma: Systemic velocity
        
    Returns
    -------
    rv_total : ndarray
        Predicted radial velocity at each time
    """
    rv_total = np.zeros_like(t, dtype=np.float64)
    
    # Planet 1
    if params_dict.get('k1', 0) != 0:
        orbel1 = np.array([
            params_dict['per1'],
            params_dict['tp1'],
            params_dict['e1'],
            params_dict['w1'],
            params_dict['k1']
        ])
        rv_total += radvel.kepler.rv_drive(t, orbel1)
    
    # Planet 2
    if params_dict.get('k2', 0) != 0:
        orbel2 = np.array([
            params_dict['per2'],
            params_dict['tp2'],
            params_dict['e2'],
            params_dict['w2'],
            params_dict['k2']
        ])
        rv_total += radvel.kepler.rv_drive(t, orbel2)
    
    # Systemic velocity
    rv_total += params_dict.get('gamma', 0.0)
    
    return rv_total

def evaluate_results(t, rv_true, rv_obs, rv_err, rv_fitted, fitted_params, true_params,
                     results_dir, t_start):
    """
    Evaluate the quality of Keplerian orbit fit and generate visualizations.
    
    Parameters
    ----------
    t : ndarray
        Observation times
    rv_true : ndarray
        True radial velocities
    rv_obs : ndarray
        Observed radial velocities
    rv_err : ndarray
        Measurement uncertainties
    rv_fitted : ndarray
        Fitted radial velocities
    fitted_params : dict
        Fitted orbital parameters
    true_params : dict
        True orbital parameters
    results_dir : str
        Directory to save results
    t_start : float
        Start time for plotting
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
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
        true_val = true_params[key]
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
        w_true = np.rad2deg(true_params[f'w{i}'])
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
    
    # Print metrics
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
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "input.npy"),
            np.column_stack([t, rv_obs, rv_err]))
    np.save(os.path.join(results_dir, "ground_truth.npy"), rv_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), rv_fitted)
    print(f"[SAVE] Input shape: ({len(t)}, 3) → input.npy")
    print(f"[SAVE] GT shape: ({len(rv_true)},) → ground_truth.npy")
    print(f"[SAVE] Recon shape: ({len(rv_fitted)},) → reconstruction.npy")
    
    # Generate visualization
    _visualize_results(t, rv_obs, rv_err, rv_true, rv_fitted, fitted_params, 
                       true_params, metrics, results_dir, t_start)
    
    return metrics

def _visualize_results(t, rv_obs, rv_err, rv_true, rv_fitted, fitted_params, 
                       true_params, metrics, results_dir, t_start):
    """Generate comprehensive visualization of RV orbit fitting."""
    residuals = rv_obs - rv_fitted
    
    # Fine time grid for smooth curves
    t_fine = np.linspace(t.min() - 5, t.max() + 5, 1000)
    rv_true_fine = forward_operator(t_fine, true_params)
    rv_fit_fine = forward_operator(t_fine, fitted_params)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Full RV time series
    ax = axes[0, 0]
    ax.errorbar(t - t_start, rv_obs, yerr=rv_err, fmt='o', ms=4, 
                color='gray', alpha=0.6, label='Observed', zorder=1)
    ax.plot(t_fine - t_start, rv_true_fine, 'b-', lw=1, alpha=0.5, label='True')
    ax.plot(t_fine - t_start, rv_fit_fine, 'r-', lw=1.5, label='Fitted')
    ax.set_xlabel('Time (days from start)')
    ax.set_ylabel('RV (m/s)')
    ax.set_title('Radial Velocity Time Series')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (b) Phase-folded Planet 1
    ax = axes[0, 1]
    # Subtract planet 2 contribution
    rv_minus_p2 = rv_obs - forward_operator(t, {**fitted_params, 'k1': 0, 'gamma': 0})
    phase1 = ((t - fitted_params['tp1']) % fitted_params['per1']) / fitted_params['per1']
    t_phase1 = np.linspace(0, 1, 200)
    t_phase1_full = fitted_params['tp1'] + t_phase1 * fitted_params['per1']
    rv_p1_model = forward_operator(t_phase1_full, {**fitted_params, 'k2': 0, 'gamma': 0})
    
    ax.errorbar(phase1, rv_minus_p2, yerr=rv_err, fmt='o', ms=4, 
                color='steelblue', alpha=0.7)
    ax.plot(t_phase1, rv_p1_model, 'r-', lw=2)
    ax.set_xlabel('Phase')
    ax.set_ylabel('RV (m/s)')
    ax.set_title(f'Planet b (P={fitted_params["per1"]:.2f} d, K={fitted_params["k1"]:.1f} m/s)')
    ax.grid(True, alpha=0.3)
    
    # (c) Phase-folded Planet 2
    ax = axes[0, 2]
    rv_minus_p1 = rv_obs - forward_operator(t, {**fitted_params, 'k2': 0, 'gamma': 0})
    phase2 = ((t - fitted_params['tp2']) % fitted_params['per2']) / fitted_params['per2']
    t_phase2 = np.linspace(0, 1, 200)
    t_phase2_full = fitted_params['tp2'] + t_phase2 * fitted_params['per2']
    rv_p2_model = forward_operator(t_phase2_full, {**fitted_params, 'k1': 0, 'gamma': 0})
    
    ax.errorbar(phase2, rv_minus_p1, yerr=rv_err, fmt='o', ms=4,
                color='darkorange', alpha=0.7)
    ax.plot(t_phase2, rv_p2_model, 'r-', lw=2)
    ax.set_xlabel('Phase')
    ax.set_ylabel('RV (m/s)')
    ax.set_title(f'Planet c (P={fitted_params["per2"]:.2f} d, K={fitted_params["k2"]:.1f} m/s)')
    ax.grid(True, alpha=0.3)
    
    # (d) Residuals
    ax = axes[1, 0]
    ax.errorbar(t - t_start, residuals, yerr=rv_err, fmt='o', ms=4,
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
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")
