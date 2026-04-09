import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

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
