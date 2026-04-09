import sys
import os
import dill
import numpy as np
import traceback
import json

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Inject the referee function (evaluate_results) from Reference B
def evaluate_results(data: dict, inversion_result: dict, output_dir: str = 'results') -> dict:
    """
    Evaluate phase retrieval results and generate metrics/visualizations.
    
    Computes RMSE, PSNR, SSIM, correlation coefficient, Strehl ratios.
    Generates visualization plots and saves results.
    
    Returns metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pupil_mask = data['pupil_mask']
    pupil_bool = data['pupil_bool']
    true_phase_rad = data['true_phase_rad']
    retrieval_nms = data['retrieval_nms']
    zernike_specs = data['zernike_specs']
    psf_infocus_noisy = data['psf_infocus_noisy']
    psf_defocus_noisy = data['psf_defocus_noisy']
    
    coefs_opt = inversion_result['coefs_opt']
    retrieved_phase_rad = inversion_result['retrieved_phase_rad']
    
    # Center phases (remove piston)
    true_mean = np.sum(true_phase_rad * pupil_mask) / np.sum(pupil_mask)
    retr_mean = np.sum(retrieved_phase_rad * pupil_mask) / np.sum(pupil_mask)
    
    true_centered = (true_phase_rad - true_mean) * pupil_mask
    retrieved_centered = (retrieved_phase_rad - retr_mean) * pupil_mask
    
    error_map = (retrieved_centered - true_centered) * pupil_mask
    
    # Compute metrics
    phase_rmse_rad = np.sqrt(np.mean(error_map[pupil_bool]**2))
    phase_rmse_waves = phase_rmse_rad / (2 * np.pi)
    
    signal_range = np.ptp(true_centered[pupil_bool])
    phase_psnr = 20 * np.log10(signal_range / phase_rmse_rad) if phase_rmse_rad > 0 else float('inf')
    
    cc = np.corrcoef(true_centered[pupil_bool], retrieved_centered[pupil_bool])[0, 1]
    
    true_rms = np.std(true_centered[pupil_bool])
    retr_rms = np.std(retrieved_centered[pupil_bool])
    strehl_true = np.exp(-true_rms**2)
    strehl_retrieved = np.exp(-retr_rms**2)
    
    # SSIM
    vmin = min(true_centered[pupil_bool].min(), retrieved_centered[pupil_bool].min())
    vmax = max(true_centered[pupil_bool].max(), retrieved_centered[pupil_bool].max())
    drange = vmax - vmin if (vmax - vmin) > 1e-10 else 1.0
    true_norm = (true_centered - vmin) / drange * pupil_mask
    retr_norm = (retrieved_centered - vmin) / drange * pupil_mask
    ssim_val = ssim(true_norm, retr_norm, data_range=1.0)
    
    # Print coefficient comparison
    print(f"\n{'Mode':<12} {'True (waves)':<14} {'Retrieved (waves)':<18} {'Error (waves)':<14}")
    print("-" * 58)
    
    truth_dict = {(n, m): c for n, m, c in zernike_specs}
    for i, (n, m) in enumerate(retrieval_nms):
        true_c = truth_dict.get((n, m), 0.0)
        retr_c = coefs_opt[i]
        err_c = retr_c - true_c
        name = f"Z({n},{m:+d})"
        if abs(true_c) > 0 or abs(retr_c) > 0.01:
            print(f"  {name:<10} {true_c:>12.4f}   {retr_c:>14.4f}     {err_c:>12.4f}")
    
    print(f"\n{'='*50}")
    print(f"Phase Retrieval Results:")
    print(f"{'='*50}")
    print(f"Phase RMSE:      {phase_rmse_rad:.4f} rad ({phase_rmse_waves:.4f} waves)")
    print(f"Phase PSNR:      {phase_psnr:.2f} dB")
    print(f"SSIM:            {ssim_val:.4f}")
    print(f"Correlation:     {cc:.6f}")
    print(f"Strehl (true):   {strehl_true:.6f}")
    print(f"Strehl (retr):   {strehl_retrieved:.6f}")
    print(f"{'='*50}")
    
    # Build metrics dictionary
    metrics = {
        'task': 'prysm_phase',
        'task_number': 126,
        'method': 'Parametric phase diversity optimization (L-BFGS-B)',
        'phase_rmse_rad': round(float(phase_rmse_rad), 4),
        'phase_rmse_waves': round(float(phase_rmse_waves), 4),
        'phase_psnr_dB': round(float(phase_psnr), 2),
        'ssim': round(float(ssim_val), 4),
        'correlation_coefficient': round(float(cc), 6),
        'strehl_ratio_true': round(float(strehl_true), 6),
        'strehl_ratio_retrieved': round(float(strehl_retrieved), 6),
        'grid_size': data['npix'],
        'n_zernike_retrieval_modes': data['n_modes'],
        'zernike_modes_truth': [{'n': n, 'm': m, 'coeff_waves': float(c)} for n, m, c in zernike_specs],
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved results/metrics.json")
    
    # Save arrays
    np.save(os.path.join(output_dir, 'ground_truth.npy'), true_centered)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), retrieved_centered)
    print("Saved results/ground_truth.npy, results/reconstruction.npy")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    phase_vmin = np.min(true_centered[pupil_bool])
    phase_vmax = np.max(true_centered[pupil_bool])
    
    # (0,0) Ground truth phase
    ax = axes[0, 0]
    disp = true_centered.copy()
    disp[~pupil_bool] = np.nan
    im = ax.imshow(disp, cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
    ax.set_title('Ground Truth Phase (rad)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Phase (rad)')
    ax.axis('off')
    
    # (0,1) Measured in-focus PSF
    ax = axes[0, 1]
    psf_d = psf_infocus_noisy.copy()
    psf_d[psf_d <= 0] = 1e-10
    im = ax.imshow(np.log10(psf_d), cmap='inferno')
    ax.set_title('Measured PSF (in-focus, log10)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='log10(I)')
    ax.axis('off')
    
    # (0,2) Measured defocused PSF
    ax = axes[0, 2]
    psf_d2 = psf_defocus_noisy.copy()
    psf_d2[psf_d2 <= 0] = 1e-10
    im = ax.imshow(np.log10(psf_d2), cmap='inferno')
    ax.set_title('Measured PSF (defocused, log10)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='log10(I)')
    ax.axis('off')
    
    # (1,0) Retrieved phase
    ax = axes[1, 0]
    disp = retrieved_centered.copy()
    disp[~pupil_bool] = np.nan
    im = ax.imshow(disp, cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
    ax.set_title('Retrieved Phase (rad)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Phase (rad)')
    ax.axis('off')
    
    # (1,1) Phase error
    ax = axes[1, 1]
    disp = error_map.copy()
    disp[~pupil_bool] = np.nan
    err_lim = max(abs(np.nanmin(disp)), abs(np.nanmax(disp)), 0.01)
    im = ax.imshow(disp, cmap='RdBu_r', vmin=-err_lim, vmax=err_lim)
    ax.set_title(f'Phase Error (RMSE={phase_rmse_rad:.4f} rad)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Error (rad)')
    ax.axis('off')
    
    # (1,2) Coefficient comparison
    ax = axes[1, 2]
    mode_labels = []
    true_vals = []
    retr_vals = []
    for i, (n, m) in enumerate(retrieval_nms):
        true_c = truth_dict.get((n, m), 0.0)
        if abs(true_c) > 0 or abs(coefs_opt[i]) > 0.005:
            mode_labels.append(f"Z({n},{m:+d})")
            true_vals.append(true_c)
            retr_vals.append(coefs_opt[i])
    
    x_pos = np.arange(len(mode_labels))
    width = 0.35
    ax.bar(x_pos - width/2, true_vals, width, label='True', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, retr_vals, width, label='Retrieved', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient (waves)', fontsize=11)
    ax.set_title('Zernike Coefficients', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Phase Retrieval: Parametric Diversity Optimization (prysm)\n'
                 f'PSNR={phase_psnr:.2f} dB | SSIM={ssim_val:.4f} | CC={cc:.4f} | '
                 f'RMSE={phase_rmse_waves:.4f} waves',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved results/reconstruction_result.png")
    
    return metrics


def extract_primary_metric(metrics):
    """
    Extract the primary scalar metric from the evaluation results.
    For phase retrieval, we use PSNR (higher is better) as the primary metric.
    """
    if isinstance(metrics, dict):
        # Use PSNR as primary metric (higher is better)
        if 'phase_psnr_dB' in metrics:
            return metrics['phase_psnr_dB'], 'higher'
        # Fallback to SSIM
        if 'ssim' in metrics:
            return metrics['ssim'], 'higher'
        # Fallback to correlation
        if 'correlation_coefficient' in metrics:
            return metrics['correlation_coefficient'], 'higher'
        # Fallback to RMSE (lower is better)
        if 'phase_rmse_rad' in metrics:
            return metrics['phase_rmse_rad'], 'lower'
    return None, None


def main():
    # Data paths provided
    data_paths = ['/data/yjh/prysm_phase_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    try:
        # Load outer data
        print("\nLoading outer data...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        # Extract args and kwargs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args length: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Determine execution pattern
        if len(inner_data_paths) > 0:
            # Chained execution pattern
            print("\n=== Chained Execution Pattern Detected ===")
            
            # Run outer function to get operator
            print("\nRunning agent run_inversion to get operator...")
            agent_operator = run_inversion(*args, **kwargs)
            
            # Load inner data and execute
            for inner_path in inner_data_paths:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_std_output = inner_data.get('output', None)
                
                # Execute the operator
                print("Executing agent operator...")
                agent_result = agent_operator(*inner_args, **inner_kwargs)
                std_result = inner_std_output
        else:
            # Direct execution pattern
            print("\n=== Direct Execution Pattern ===")
            
            # Run the inversion
            print("\nRunning agent run_inversion...")
            agent_result = run_inversion(*args, **kwargs)
            std_result = std_output
        
        print(f"\nAgent result type: {type(agent_result)}")
        print(f"Standard result type: {type(std_result)}")
        
        if isinstance(agent_result, dict):
            print(f"Agent result keys: {agent_result.keys()}")
        if isinstance(std_result, dict):
            print(f"Standard result keys: {std_result.keys()}")
        
        # For evaluation, we need the input data which should be the first argument
        # The function signature is run_inversion(data: dict)
        if len(args) > 0:
            input_data = args[0]
        elif 'data' in kwargs:
            input_data = kwargs['data']
        else:
            print("ERROR: Could not find input data for evaluation!")
            sys.exit(1)
        
        # Evaluate both results
        print("\n" + "="*60)
        print("EVALUATING AGENT RESULT")
        print("="*60)
        agent_metrics = evaluate_results(input_data, agent_result, output_dir='results_agent')
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD RESULT")
        print("="*60)
        std_metrics = evaluate_results(input_data, std_result, output_dir='results_std')
        
        # Extract primary metrics
        agent_score, metric_type = extract_primary_metric(agent_metrics)
        std_score, _ = extract_primary_metric(std_metrics)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Primary Metric: PSNR (dB)")
        print(f"Scores -> Agent: {agent_score}, Standard: {std_score}")
        print(f"Metric type: {metric_type} is better")
        
        # Additional metrics comparison
        print("\nDetailed Metrics Comparison:")
        print(f"  PSNR (dB):    Agent={agent_metrics.get('phase_psnr_dB', 'N/A')}, Std={std_metrics.get('phase_psnr_dB', 'N/A')}")
        print(f"  SSIM:         Agent={agent_metrics.get('ssim', 'N/A')}, Std={std_metrics.get('ssim', 'N/A')}")
        print(f"  Correlation:  Agent={agent_metrics.get('correlation_coefficient', 'N/A')}, Std={std_metrics.get('correlation_coefficient', 'N/A')}")
        print(f"  RMSE (rad):   Agent={agent_metrics.get('phase_rmse_rad', 'N/A')}, Std={std_metrics.get('phase_rmse_rad', 'N/A')}")
        
        # Determine success with tolerance
        # For optimization algorithms, we allow some variation due to:
        # 1. Different random seeds in multi-start optimization
        # 2. Numerical precision differences
        # 3. Convergence tolerance
        
        tolerance = 0.15  # 15% tolerance for optimization algorithms
        
        if agent_score is None or std_score is None:
            print("\nWARNING: Could not extract primary metric for comparison")
            # Fall back to checking if agent result is reasonable
            if agent_metrics.get('phase_psnr_dB', 0) > 10:  # Reasonable PSNR
                print("Agent result appears reasonable (PSNR > 10 dB)")
                sys.exit(0)
            else:
                print("Agent result appears poor")
                sys.exit(1)
        
        if metric_type == 'higher':
            # Higher is better (PSNR, SSIM, correlation)
            threshold = std_score * (1 - tolerance)
            passed = agent_score >= threshold
            print(f"\nThreshold (with {tolerance*100}% tolerance): {threshold:.4f}")
            print(f"Agent score: {agent_score:.4f}")
        else:
            # Lower is better (RMSE)
            threshold = std_score * (1 + tolerance)
            passed = agent_score <= threshold
            print(f"\nThreshold (with {tolerance*100}% tolerance): {threshold:.4f}")
            print(f"Agent score: {agent_score:.4f}")
        
        if passed:
            print("\n✓ TEST PASSED: Agent performance is acceptable")
            sys.exit(0)
        else:
            print("\n✗ TEST FAILED: Agent performance degraded significantly")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()