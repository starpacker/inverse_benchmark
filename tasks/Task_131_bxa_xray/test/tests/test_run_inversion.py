import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the agent's function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def photoelectric_cross_section(E_keV):
    """Approximate photoelectric absorption cross-section in cm^2.
    Simplified Morrison & McCammon approximation."""
    return 2.0e-22 * (E_keV) ** (-8.0/3.0)

def absorbed_powerlaw(E_keV, gamma, K, N_H_1e22):
    """Absorbed power-law X-ray spectrum."""
    sigma = photoelectric_cross_section(E_keV)
    absorption = np.exp(-N_H_1e22 * 1e22 * sigma)
    return K * E_keV**(-gamma) * absorption

def evaluate_results(data_dict, result_dict):
    """Evaluate reconstruction quality and save results."""
    E = data_dict['E_centers']
    dE = data_dict['dE']
    observed = data_dict['observed']
    expected = data_dict['expected_counts']
    background = data_dict['background']
    true_params = data_dict['true_params']
    
    gamma_fit = result_dict['gamma_fit']
    K_fit = result_dict['K_fit']
    NH_fit = result_dict['NH_fit']
    recovered = result_dict['recovered_counts']
    
    # True total counts (source + background)
    true_total = expected + background
    
    # Compute metrics
    psnr_val = 10 * np.log10(np.max(true_total)**2 / np.mean((true_total - recovered)**2))
    cc = np.corrcoef(true_total, recovered)[0, 1]
    rmse = np.sqrt(np.mean((true_total - recovered)**2))
    
    # Parameter errors
    param_errors = {
        'gamma': {
            'true': true_params['gamma'], 
            'fitted': gamma_fit,
            'rel_error': abs(gamma_fit - true_params['gamma']) / true_params['gamma']
        },
        'K': {
            'true': true_params['K'], 
            'fitted': K_fit,
            'rel_error': abs(K_fit - true_params['K']) / true_params['K']
        },
        'N_H': {
            'true': true_params['N_H'], 
            'fitted': NH_fit,
            'rel_error': abs(NH_fit - true_params['N_H']) / true_params['N_H']
        }
    }
    mean_rel_error = np.mean([v['rel_error'] for v in param_errors.values()])
    
    # Build metrics dictionary
    metrics = {
        'task': 'bxa_xray',
        'psnr_db': float(psnr_val),
        'correlation_coefficient': float(cc),
        'rmse_counts': float(rmse),
        'mean_parameter_relative_error': float(mean_rel_error),
        'parameters': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in param_errors.items()}
    }
    
    # Print evaluation results
    print(f"[EVAL] PSNR = {psnr_val:.2f} dB, CC = {cc:.6f}, RMSE = {rmse:.2f}")
    for k, v in param_errors.items():
        print(f"  {k}: true={v['true']}, fitted={v['fitted']:.6f}, error={v['rel_error']*100:.2f}%")
    
    return metrics

# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/bxa_xray_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # ---- Load outer (primary) data ----
    assert len(outer_files) == 1, f"Expected exactly 1 outer file, got {len(outer_files)}"
    with open(outer_files[0], 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Outer func_name: {outer_data.get('func_name', 'N/A')}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    if len(inner_files) > 0:
        # ---- Pattern 2: Chained Execution ----
        print("\n[INFO] Pattern 2: Chained Execution detected.")
        
        # Step 1: Run outer function to get operator
        agent_operator = run_inversion(*args, **kwargs)
        
        # Step 2: Load inner data and run operator
        with open(inner_files[0], 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        agent_result = agent_operator(*inner_args, **inner_kwargs)
        
        # The data_dict for evaluation is the input to run_inversion
        data_dict = args[0] if len(args) > 0 else kwargs.get('data_dict', None)
    else:
        # ---- Pattern 1: Direct Execution ----
        print("\n[INFO] Pattern 1: Direct Execution detected.")
        
        # Run agent's function
        agent_result = run_inversion(*args, **kwargs)
        std_result = std_output
        
        # The data_dict for evaluation is the input to run_inversion
        data_dict = args[0] if len(args) > 0 else kwargs.get('data_dict', None)
    
    # ---- Sanity checks ----
    assert agent_result is not None, "Agent returned None"
    assert std_result is not None, "Standard result is None"
    assert data_dict is not None, "Could not extract data_dict for evaluation"
    
    print(f"\nAgent result keys: {list(agent_result.keys()) if isinstance(agent_result, dict) else type(agent_result)}")
    print(f"Std result keys: {list(std_result.keys()) if isinstance(std_result, dict) else type(std_result)}")
    
    # ---- Evaluation Phase ----
    print("\n" + "="*60)
    print("Evaluating AGENT result:")
    print("="*60)
    try:
        metrics_agent = evaluate_results(data_dict, agent_result)
    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Evaluating STANDARD result:")
    print("="*60)
    try:
        metrics_std = evaluate_results(data_dict, std_result)
    except Exception as e:
        print(f"[ERROR] Standard evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ---- Extract primary metrics ----
    psnr_agent = metrics_agent['psnr_db']
    psnr_std = metrics_std['psnr_db']
    cc_agent = metrics_agent['correlation_coefficient']
    cc_std = metrics_std['correlation_coefficient']
    rmse_agent = metrics_agent['rmse_counts']
    rmse_std = metrics_std['rmse_counts']
    mre_agent = metrics_agent['mean_parameter_relative_error']
    mre_std = metrics_std['mean_parameter_relative_error']
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"  PSNR (dB)    -> Agent: {psnr_agent:.2f}, Standard: {psnr_std:.2f}")
    print(f"  Correlation  -> Agent: {cc_agent:.6f}, Standard: {cc_std:.6f}")
    print(f"  RMSE         -> Agent: {rmse_agent:.2f}, Standard: {rmse_std:.2f}")
    print(f"  Mean Rel Err -> Agent: {mre_agent*100:.2f}%, Standard: {mre_std*100:.2f}%")
    
    # ---- Verification ----
    # PSNR: higher is better. Allow 10% margin.
    # CC: higher is better. Allow small margin.
    # RMSE: lower is better.
    # Mean relative error: lower is better.
    
    passed = True
    
    # Check PSNR (higher is better) - allow 10% degradation
    if psnr_agent < psnr_std * 0.9:
        print(f"\n[FAIL] PSNR degraded significantly: {psnr_agent:.2f} < {psnr_std * 0.9:.2f} (90% of standard)")
        passed = False
    else:
        print(f"\n[PASS] PSNR acceptable: {psnr_agent:.2f} >= {psnr_std * 0.9:.2f}")
    
    # Check CC (higher is better) - allow small margin
    if cc_agent < cc_std * 0.95:
        print(f"[FAIL] Correlation degraded: {cc_agent:.6f} < {cc_std * 0.95:.6f}")
        passed = False
    else:
        print(f"[PASS] Correlation acceptable: {cc_agent:.6f} >= {cc_std * 0.95:.6f}")
    
    # Check RMSE (lower is better) - allow 10% margin
    if rmse_agent > rmse_std * 1.1:
        print(f"[FAIL] RMSE degraded: {rmse_agent:.2f} > {rmse_std * 1.1:.2f}")
        passed = False
    else:
        print(f"[PASS] RMSE acceptable: {rmse_agent:.2f} <= {rmse_std * 1.1:.2f}")
    
    # Check mean relative error (lower is better) - allow 10% margin
    if mre_agent > mre_std * 1.1 + 0.01:  # small absolute margin too
        print(f"[FAIL] Mean parameter error degraded: {mre_agent*100:.2f}% > threshold")
        passed = False
    else:
        print(f"[PASS] Mean parameter error acceptable: {mre_agent*100:.2f}%")
    
    # Check optimization success
    if isinstance(agent_result, dict) and not agent_result.get('optimization_success', True):
        print("[WARN] Agent optimization did not report success")
    
    print("\n" + "="*60)
    if passed:
        print("OVERALL: PASSED - Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("OVERALL: FAILED - Agent performance degraded significantly.")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)