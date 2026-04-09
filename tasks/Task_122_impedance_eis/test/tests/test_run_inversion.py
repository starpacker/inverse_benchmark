import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the evaluate_results function (Reference B) verbatim
# ============================================================
def evaluate_results(data, inversion_result, output_dir='results'):
    """
    Compute metrics, save results, and generate visualizations.
    """
    freq = data['freq']
    Z_true = data['Z_true']
    Z_noisy = data['Z_noisy']
    gt_vec = data['gt_vec']
    param_order = data['param_order']
    noise_level = data['noise_level']
    
    fitted_params = inversion_result['fitted_params']
    Z_fitted = inversion_result['Z_fitted']
    converged = inversion_result['converged']
    
    # Parameter relative errors
    param_errors = {}
    for i, name in enumerate(param_order):
        gt_val = gt_vec[i]
        fit_val = fitted_params[i]
        rel_err = np.abs(fit_val - gt_val) / np.abs(gt_val)
        param_errors[name] = {
            'ground_truth': float(gt_val),
            'fitted': float(fit_val),
            'relative_error': float(rel_err),
        }
        print(f"  {name}: GT={gt_val:.4e}, Fit={fit_val:.4e}, RelErr={rel_err:.4f}")
    
    # Spectral RMSE (on complex impedance)
    rmse = np.sqrt(np.mean(np.abs(Z_fitted - Z_true)**2))
    
    # Spectral PSNR
    max_Z = np.max(np.abs(Z_true))
    psnr = 20.0 * np.log10(max_Z / rmse) if rmse > 0 else float('inf')
    
    # Correlation coefficient between fitted and true spectra
    sig_true = np.concatenate([Z_true.real, Z_true.imag])
    sig_fit = np.concatenate([Z_fitted.real, Z_fitted.imag])
    cc = float(np.corrcoef(sig_true, sig_fit)[0, 1])
    
    # Mean parameter relative error
    mean_re = float(np.mean([v['relative_error'] for v in param_errors.values()]))
    
    print(f"\nSpectral RMSE: {rmse:.4f} Ohm")
    print(f"Spectral PSNR: {psnr:.2f} dB")
    print(f"Correlation Coefficient: {cc:.6f}")
    print(f"Mean Parameter Relative Error: {mean_re:.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build metrics dictionary
    metrics = {
        'task': 'impedance_eis',
        'task_number': 122,
        'description': 'Electrochemical Impedance Spectroscopy (EIS) Randles circuit fitting',
        'psnr_db': round(float(psnr), 2),
        'rmse_ohm': round(float(rmse), 4),
        'correlation_coefficient': round(cc, 6),
        'mean_parameter_relative_error': round(mean_re, 6),
        'parameters': param_errors,
        'noise_level_percent': noise_level * 100,
        'num_frequencies': len(freq),
        'optimizer': 'L-BFGS-B',
        'converged': converged,
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save spectra as npy
    np.save(os.path.join(output_dir, 'ground_truth.npy'), 
            np.stack([Z_true.real, Z_true.imag]))
    np.save(os.path.join(output_dir, 'reconstruction.npy'), 
            np.stack([Z_fitted.real, Z_fitted.imag]))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Task 122: EIS Randles Circuit Fitting', fontsize=15, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(Z_true.real, -Z_true.imag, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(Z_noisy.real, -Z_noisy.imag, 'k.', markersize=5, alpha=0.5, label='Noisy Data')
    ax.plot(Z_fitted.real, -Z_fitted.imag, 'r--', linewidth=2, label='Fitted')
    ax.set_xlabel('Z_real (Ω)', fontsize=11)
    ax.set_ylabel('-Z_imag (Ω)', fontsize=11)
    ax.set_title('(a) Nyquist Plot', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.loglog(freq, np.abs(Z_true), 'b-', linewidth=2, label='|Z| GT')
    ax.loglog(freq, np.abs(Z_noisy), 'k.', markersize=4, alpha=0.4, label='|Z| Noisy')
    ax.loglog(freq, np.abs(Z_fitted), 'r--', linewidth=2, label='|Z| Fitted')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('|Z| (Ω)', fontsize=11)
    ax.set_title('(b) Bode Plot – Magnitude', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    ax2 = ax.twinx()
    phase_true = np.degrees(np.angle(Z_true))
    phase_fitted = np.degrees(np.angle(Z_fitted))
    ax2.semilogx(freq, phase_true, 'b:', linewidth=1.5, alpha=0.6)
    ax2.semilogx(freq, phase_fitted, 'r:', linewidth=1.5, alpha=0.6)
    ax2.set_ylabel('Phase (°)', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax = axes[1, 0]
    names = param_order
    gt_vals_norm = []
    fit_vals_norm = []
    for i, name in enumerate(names):
        gt_v = gt_vec[i]
        fit_v = fitted_params[i]
        gt_vals_norm.append(1.0)
        fit_vals_norm.append(fit_v / gt_v)
    
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, gt_vals_norm, width, label='Ground Truth', color='steelblue')
    ax.bar(x_pos + width/2, fit_vals_norm, width, label='Fitted', color='salmon')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Normalised Value (GT = 1)', fontsize=11)
    ax.set_title('(c) Parameter Recovery (normalised)', fontsize=12)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, name in enumerate(names):
        re = param_errors[name]['relative_error']
        ax.text(x_pos[i] + width/2, fit_vals_norm[i] + 0.02,
                f'{re:.1%}', ha='center', va='bottom', fontsize=8, color='red')
    
    ax = axes[1, 1]
    residual_re = Z_fitted.real - Z_true.real
    residual_im = Z_fitted.imag - Z_true.imag
    ax.semilogx(freq, residual_re, 'b.-', markersize=4, label='ΔZ_real', alpha=0.7)
    ax.semilogx(freq, residual_im, 'r.-', markersize=4, label='ΔZ_imag', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Residual (Ω)', fontsize=11)
    ax.set_title('(d) Fit Residuals (Fitted − GT)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All results saved to {output_dir}/")
    print(f"  metrics.json, reconstruction_result.png, ground_truth.npy, reconstruction.npy")
    
    return metrics


# ============================================================
# Main test logic
# ============================================================
def main():
    data_paths = ['/data/yjh/impedance_eis_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's run_inversion
    print("\n===== Running agent's run_inversion =====")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if chained execution
    if len(inner_paths) > 0:
        print(f"\nChained execution detected. Inner files: {inner_paths}")
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print("Running agent_output as callable...")
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Inner call failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    # Extract the data dict (first arg) for evaluate_results
    data_dict = args[0] if len(args) > 0 else kwargs.get('data', None)
    
    if data_dict is None:
        print("ERROR: Could not extract data dictionary for evaluation.")
        sys.exit(1)
    
    # Evaluate agent result
    print("\n===== Evaluating Agent Result =====")
    try:
        agent_metrics = evaluate_results(data_dict, final_result, output_dir='results_agent')
    except Exception as e:
        print(f"ERROR: evaluate_results failed for agent: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("\n===== Evaluating Standard Result =====")
    try:
        std_metrics = evaluate_results(data_dict, std_result, output_dir='results_std')
    except Exception as e:
        print(f"ERROR: evaluate_results failed for standard: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract key metrics for comparison
    agent_psnr = agent_metrics['psnr_db']
    std_psnr = std_metrics['psnr_db']
    agent_rmse = agent_metrics['rmse_ohm']
    std_rmse = std_metrics['rmse_ohm']
    agent_cc = agent_metrics['correlation_coefficient']
    std_cc = std_metrics['correlation_coefficient']
    agent_mean_re = agent_metrics['mean_parameter_relative_error']
    std_mean_re = std_metrics['mean_parameter_relative_error']
    
    print("\n===== Comparison =====")
    print(f"PSNR (dB)     -> Agent: {agent_psnr}, Standard: {std_psnr}")
    print(f"RMSE (Ohm)    -> Agent: {agent_rmse}, Standard: {std_rmse}")
    print(f"Correlation   -> Agent: {agent_cc}, Standard: {std_cc}")
    print(f"Mean Rel Err  -> Agent: {agent_mean_re}, Standard: {std_mean_re}")
    
    # Determine pass/fail
    # PSNR: higher is better (allow 10% margin)
    # RMSE: lower is better (allow 10% margin)
    # Correlation: higher is better (allow small margin)
    # Mean Rel Error: lower is better (allow 10% margin)
    
    passed = True
    
    # PSNR check (higher is better)
    if agent_psnr < std_psnr * 0.9:
        print(f"FAIL: Agent PSNR ({agent_psnr}) is significantly worse than Standard ({std_psnr})")
        passed = False
    else:
        print(f"PASS: PSNR check passed")
    
    # RMSE check (lower is better)
    if agent_rmse > std_rmse * 1.1:
        print(f"FAIL: Agent RMSE ({agent_rmse}) is significantly worse than Standard ({std_rmse})")
        passed = False
    else:
        print(f"PASS: RMSE check passed")
    
    # Correlation check
    if agent_cc < std_cc * 0.99:
        print(f"FAIL: Agent Correlation ({agent_cc}) is significantly worse than Standard ({std_cc})")
        passed = False
    else:
        print(f"PASS: Correlation check passed")
    
    # Mean relative error check (lower is better)
    if std_mean_re > 0 and agent_mean_re > std_mean_re * 1.5:
        print(f"FAIL: Agent Mean Rel Error ({agent_mean_re}) is significantly worse than Standard ({std_mean_re})")
        passed = False
    else:
        print(f"PASS: Mean Relative Error check passed")
    
    if passed:
        print("\n✓ All checks PASSED. Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("\n✗ Some checks FAILED. Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == '__main__':
    main()