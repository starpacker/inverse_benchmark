import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import the target function
from agent_run_inversion import run_inversion

# Import radvel for forward_operator
import radvel

# Inject the forward_operator and evaluate_results from Reference B
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
        true_val = true_params.get(key, 0)
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
        w_true = np.rad2deg(true_params.get(f'w{i}', 0))
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
    print(f"[EVAL] Per1 rel error = {metrics.get('per1_rel_error', 0)*100:.4f}%")
    print(f"[EVAL] Per2 rel error = {metrics.get('per2_rel_error', 0)*100:.4f}%")
    print(f"[EVAL] K1 rel error = {metrics.get('k1_rel_error', 0)*100:.4f}%")
    print(f"[EVAL] K2 rel error = {metrics.get('k2_rel_error', 0)*100:.4f}%")
    print(f"[EVAL] e1 rel error = {metrics.get('e1_rel_error', 0)*100:.4f}%")
    print(f"[EVAL] e2 rel error = {metrics.get('e2_rel_error', 0)*100:.4f}%")
    print(f"[EVAL] ω1 error = {metrics.get('w1_error_deg', 0):.4f}°")
    print(f"[EVAL] ω2 error = {metrics.get('w2_error_deg', 0):.4f}°")
    
    # Save metrics
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
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
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/radvel_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    # Load outer data
    if not outer_files:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"[INFO] Loading outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Function: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Args count: {len(args)}")
    print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
    
    # Extract input data
    t = args[0] if len(args) > 0 else kwargs.get('t')
    rv_obs = args[1] if len(args) > 1 else kwargs.get('rv_obs')
    rv_err = args[2] if len(args) > 2 else kwargs.get('rv_err')
    
    print(f"[INFO] t shape: {t.shape if hasattr(t, 'shape') else len(t)}")
    print(f"[INFO] rv_obs shape: {rv_obs.shape if hasattr(rv_obs, 'shape') else len(rv_obs)}")
    print(f"[INFO] rv_err shape: {rv_err.shape if hasattr(rv_err, 'shape') else len(rv_err)}")
    
    # Run the agent's run_inversion
    print("\n[INFO] Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner files (chained execution)
    if inner_files:
        print("[INFO] Chained execution detected - loading inner data...")
        inner_path = inner_files[0]
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        if callable(agent_output):
            print("[INFO] Executing returned operator...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            final_result = agent_output
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    # Parse outputs
    # Both agent and std outputs should be (fitted_params, rv_fitted)
    if isinstance(final_result, tuple) and len(final_result) == 2:
        agent_fitted_params, agent_rv_fitted = final_result
    else:
        print(f"[ERROR] Unexpected agent output format: {type(final_result)}")
        sys.exit(1)
    
    if isinstance(std_result, tuple) and len(std_result) == 2:
        std_fitted_params, std_rv_fitted = std_result
    else:
        print(f"[ERROR] Unexpected standard output format: {type(std_result)}")
        sys.exit(1)
    
    print(f"\n[INFO] Agent fitted params keys: {list(agent_fitted_params.keys())}")
    print(f"[INFO] Standard fitted params keys: {list(std_fitted_params.keys())}")
    
    # Create true params from standard result (use as ground truth proxy)
    # In real scenario, true_params would be provided separately
    # Here we use std_fitted_params as the reference
    true_params = std_fitted_params.copy()
    
    # Compute true RV using forward operator with true params
    rv_true = forward_operator(t, true_params)
    
    # Setup results directory
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    t_start = t.min()
    
    # Evaluate agent's result
    print("\n" + "="*60)
    print("[EVAL] Evaluating Agent's Result:")
    print("="*60)
    agent_metrics = evaluate_results(
        t, rv_true, rv_obs, rv_err, agent_rv_fitted, 
        agent_fitted_params, true_params, 
        os.path.join(results_dir, "agent"), t_start
    )
    
    # Evaluate standard result
    print("\n" + "="*60)
    print("[EVAL] Evaluating Standard Result:")
    print("="*60)
    std_metrics = evaluate_results(
        t, rv_true, rv_obs, rv_err, std_rv_fitted, 
        std_fitted_params, true_params, 
        os.path.join(results_dir, "standard"), t_start
    )
    
    # Compare scores
    print("\n" + "="*60)
    print("[COMPARE] Score Comparison:")
    print("="*60)
    
    agent_psnr = agent_metrics['psnr']
    std_psnr = std_metrics['psnr']
    agent_cc = agent_metrics['cc']
    std_cc = std_metrics['cc']
    agent_rms = agent_metrics['rms_residuals']
    std_rms = std_metrics['rms_residuals']
    
    print(f"Scores -> Agent PSNR: {agent_psnr:.4f}, Standard PSNR: {std_psnr:.4f}")
    print(f"Scores -> Agent CC: {agent_cc:.6f}, Standard CC: {std_cc:.6f}")
    print(f"Scores -> Agent RMS: {agent_rms:.4f}, Standard RMS: {std_rms:.4f}")
    
    # Determine success
    # PSNR: Higher is better
    # CC: Higher is better (closer to 1)
    # RMS: Lower is better
    
    # Allow 10% margin for PSNR and CC, 20% margin for RMS
    psnr_ok = agent_psnr >= std_psnr * 0.9 or agent_psnr >= std_psnr - 2.0  # Within 2dB
    cc_ok = agent_cc >= std_cc * 0.95 or agent_cc >= 0.99  # Very high correlation is fine
    rms_ok = agent_rms <= std_rms * 1.2  # Allow 20% worse RMS
    
    print(f"\n[CHECK] PSNR OK: {psnr_ok} (agent >= std * 0.9 or within 2dB)")
    print(f"[CHECK] CC OK: {cc_ok} (agent >= std * 0.95 or >= 0.99)")
    print(f"[CHECK] RMS OK: {rms_ok} (agent <= std * 1.2)")
    
    # Final verdict
    if psnr_ok and cc_ok and rms_ok:
        print("\n[RESULT] ✓ PASS - Agent performance is acceptable")
        sys.exit(0)
    else:
        print("\n[RESULT] ✗ FAIL - Agent performance degraded significantly")
        sys.exit(1)


if __name__ == "__main__":
    main()