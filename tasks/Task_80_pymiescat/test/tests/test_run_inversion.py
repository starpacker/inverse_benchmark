import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Inject the evaluate_results function (Reference B)
def evaluate_results(ground_truth, recon_params, recon_spectra, observations, results_dir):
    """
    Compute comprehensive evaluation metrics and generate visualizations.
    
    Args:
        ground_truth: dict with true parameters and clean spectra
        recon_params: dict with recovered n_real, k_imag, diameter
        recon_spectra: dict with reconstructed qsca, qabs, qext, g arrays
        observations: dict with wavelengths and noisy observations
        results_dir: directory path to save results
    
    Returns:
        metrics: dict with all evaluation metrics
    """
    # Parameter recovery errors
    n_error = abs(recon_params['n_real'] - ground_truth['n_real'])
    k_error = abs(recon_params['k_imag'] - ground_truth['k_imag'])
    d_error = abs(recon_params['diameter'] - ground_truth['diameter'])
    
    n_re = n_error / ground_truth['n_real']
    k_re = k_error / ground_truth['k_imag']
    d_re = d_error / ground_truth['diameter']
    
    # Spectral reconstruction quality
    gt_qsca = ground_truth['qsca_clean']
    gt_qabs = ground_truth['qabs_clean']
    gt_qext = ground_truth['qext_clean']
    
    rec_qsca = recon_spectra['qsca']
    rec_qabs = recon_spectra['qabs']
    rec_qext = recon_spectra['qext']
    
    # RMSE for spectra
    rmse_qsca = np.sqrt(np.mean((gt_qsca - rec_qsca)**2))
    rmse_qabs = np.sqrt(np.mean((gt_qabs - rec_qabs)**2))
    rmse_qext = np.sqrt(np.mean((gt_qext - rec_qext)**2))
    
    # Correlation coefficients
    cc_qsca = np.corrcoef(gt_qsca, rec_qsca)[0, 1]
    cc_qabs = np.corrcoef(gt_qabs, rec_qabs)[0, 1]
    cc_qext = np.corrcoef(gt_qext, rec_qext)[0, 1]
    
    # PSNR for Qext spectrum
    data_range = gt_qext.max() - gt_qext.min()
    mse_qext = np.mean((gt_qext - rec_qext)**2)
    psnr = 10 * np.log10(data_range**2 / mse_qext) if mse_qext > 0 else float('inf')
    
    # Overall relative error
    gt_all = np.concatenate([gt_qsca, gt_qabs])
    rec_all = np.concatenate([rec_qsca, rec_qabs])
    overall_re = np.sqrt(np.mean((gt_all - rec_all)**2)) / np.sqrt(np.mean(gt_all**2))
    
    metrics = {
        'n_real_gt': ground_truth['n_real'],
        'n_real_recon': recon_params['n_real'],
        'n_real_error': float(n_error),
        'n_real_relative_error': float(n_re),
        'k_imag_gt': ground_truth['k_imag'],
        'k_imag_recon': recon_params['k_imag'],
        'k_imag_error': float(k_error),
        'k_imag_relative_error': float(k_re),
        'diameter_gt': ground_truth['diameter'],
        'diameter_recon': recon_params['diameter'],
        'diameter_error': float(d_error),
        'diameter_relative_error': float(d_re),
        'rmse_qsca': float(rmse_qsca),
        'rmse_qabs': float(rmse_qabs),
        'rmse_qext': float(rmse_qext),
        'cc_qsca': float(cc_qsca),
        'cc_qabs': float(cc_qabs),
        'cc_qext': float(cc_qext),
        'psnr': float(psnr),
        'rmse': float(rmse_qext),
        'overall_relative_error': float(overall_re),
    }
    
    # Print metrics
    print(f"[EVAL] n error = {metrics['n_real_error']:.6f} "
          f"(RE: {metrics['n_real_relative_error']:.6f})")
    print(f"[EVAL] k error = {metrics['k_imag_error']:.6f} "
          f"(RE: {metrics['k_imag_relative_error']:.6f})")
    print(f"[EVAL] d error = {metrics['diameter_error']:.2f} nm "
          f"(RE: {metrics['diameter_relative_error']:.6f})")
    print(f"[EVAL] CC_Qsca = {metrics['cc_qsca']:.6f}")
    print(f"[EVAL] CC_Qabs = {metrics['cc_qabs']:.6f}")
    print(f"[EVAL] CC_Qext = {metrics['cc_qext']:.6f}")
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] Overall RE = {metrics['overall_relative_error']:.6f}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    gt_spectra = np.stack([ground_truth['qsca_clean'], 
                           ground_truth['qabs_clean'],
                           ground_truth['qext_clean']], axis=0)
    rec_spectra_arr = np.stack([recon_spectra['qsca'],
                                recon_spectra['qabs'],
                                recon_spectra['qext']], axis=0)
    input_data = np.stack([observations['qsca'],
                           observations['qabs'],
                           observations['qext']], axis=0)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_spectra)
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_spectra_arr)
    np.save(os.path.join(results_dir, "input.npy"), input_data)
    print(f"[SAVE] GT spectra shape: {gt_spectra.shape} → ground_truth.npy")
    print(f"[SAVE] Recon spectra shape: {rec_spectra_arr.shape} → reconstruction.npy")
    print(f"[SAVE] Input spectra shape: {input_data.shape} → input.npy")
    
    # Generate visualization
    wl = observations['wavelengths']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Qsca spectrum comparison
    axes[0, 0].plot(wl, ground_truth['qsca_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 0].plot(wl, observations['qsca'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 0].plot(wl, recon_spectra['qsca'], 'r--', lw=2, label='Reconstructed')
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Qsca')
    axes[0, 0].set_title(f'Scattering Efficiency (CC={metrics["cc_qsca"]:.6f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) Qabs spectrum comparison
    axes[0, 1].plot(wl, ground_truth['qabs_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 1].plot(wl, observations['qabs'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 1].plot(wl, recon_spectra['qabs'], 'r--', lw=2, label='Reconstructed')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Qabs')
    axes[0, 1].set_title(f'Absorption Efficiency (CC={metrics["cc_qabs"]:.6f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) Qext spectrum comparison
    axes[0, 2].plot(wl, ground_truth['qext_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 2].plot(wl, observations['qext'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 2].plot(wl, recon_spectra['qext'], 'r--', lw=2, label='Reconstructed')
    axes[0, 2].set_xlabel('Wavelength (nm)')
    axes[0, 2].set_ylabel('Qext')
    axes[0, 2].set_title(f'Extinction Efficiency (CC={metrics["cc_qext"]:.6f})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # (d) Residuals
    res_sca = ground_truth['qsca_clean'] - recon_spectra['qsca']
    res_abs = ground_truth['qabs_clean'] - recon_spectra['qabs']
    axes[1, 0].plot(wl, res_sca, 'b-', lw=1.5, label='Qsca residual')
    axes[1, 0].plot(wl, res_abs, 'r-', lw=1.5, label='Qabs residual')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('Spectral Residuals (GT - Recon)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (e) Parameter comparison bar chart
    params_names = ['n (real)', 'k (imag)', 'd (nm)']
    gt_vals = [ground_truth['n_real'], ground_truth['k_imag'], ground_truth['diameter']]
    rec_vals = [recon_params['n_real'], recon_params['k_imag'], recon_params['diameter']]
    
    # Normalize for display
    gt_norm = np.array(gt_vals) / np.array(gt_vals)
    rec_norm = np.array(rec_vals) / np.array(gt_vals)
    
    x_pos = np.arange(len(params_names))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, gt_norm, width, label='Ground Truth', color='steelblue')
    axes[1, 1].bar(x_pos + width/2, rec_norm, width, label='Reconstructed', color='coral')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(params_names)
    axes[1, 1].set_ylabel('Normalized Value (GT=1.0)')
    axes[1, 1].set_title('Parameter Recovery')
    axes[1, 1].legend()
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add error annotations
    for i, (name, gt, rec) in enumerate(zip(params_names, gt_vals, rec_vals)):
        re = abs(rec - gt) / gt * 100
        axes[1, 1].annotate(f'{re:.2f}%', xy=(i, max(gt_norm[i], rec_norm[i]) + 0.02),
                            ha='center', fontsize=9, color='red')
    
    # (f) Asymmetry parameter
    axes[1, 2].plot(wl, ground_truth['g_clean'], 'b-', lw=2, label='GT g(λ)')
    axes[1, 2].plot(wl, recon_spectra['g'], 'r--', lw=2, label='Recon g(λ)')
    axes[1, 2].set_xlabel('Wavelength (nm)')
    axes[1, 2].set_ylabel('g (asymmetry parameter)')
    axes[1, 2].set_title('Asymmetry Parameter')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"PyMieScatt — Mie Scattering Refractive Index Inversion\n"
        f"GT: n={ground_truth['n_real']}, k={ground_truth['k_imag']}, d={ground_truth['diameter']} nm | "
        f"Recon: n={recon_params['n_real']:.4f}, k={recon_params['k_imag']:.6f}, d={recon_params['diameter']:.2f} nm\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC_ext={metrics['cc_qext']:.6f} | RE={metrics['overall_relative_error']:.6f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/pymiescat_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    print(f"[INFO] Outer data: {outer_data_path}")
    print(f"[INFO] Inner data: {inner_data_paths}")
    
    try:
        # Load outer data
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Loaded outer data with keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Running agent's run_inversion with {len(args)} args and {len(kwargs)} kwargs")
        
        # Run the agent's function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_data_paths:
            # Chained execution pattern
            print("[INFO] Detected chained execution pattern")
            inner_data_path = inner_data_paths[0]
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("[INFO] Direct execution pattern")
            final_result = agent_output
            std_result = std_output
        
        # Both final_result and std_result should be tuples: (recon_params, recon_spectra)
        print(f"[INFO] Agent result type: {type(final_result)}")
        print(f"[INFO] Standard result type: {type(std_result)}")
        
        # Extract recon_params and recon_spectra from both results
        if isinstance(final_result, tuple) and len(final_result) == 2:
            agent_recon_params, agent_recon_spectra = final_result
        else:
            print("[ERROR] Unexpected agent output format")
            sys.exit(1)
        
        if isinstance(std_result, tuple) and len(std_result) == 2:
            std_recon_params, std_recon_spectra = std_result
        else:
            print("[ERROR] Unexpected standard output format")
            sys.exit(1)
        
        # For evaluation, we need ground_truth and observations
        # These should be in the input args
        # observations is the first argument to run_inversion
        observations = args[0] if args else kwargs.get('observations', None)
        
        if observations is None:
            print("[ERROR] Could not find observations in input args")
            sys.exit(1)
        
        # We need to construct ground_truth from observations or use a known structure
        # Looking at the evaluate_results function, it expects:
        # ground_truth: dict with n_real, k_imag, diameter, qsca_clean, qabs_clean, qext_clean, g_clean
        
        # Since we're comparing agent vs standard, we can use the standard result as a proxy
        # The standard output IS the "ground truth" for comparison purposes
        # But evaluate_results expects actual ground truth parameters
        
        # For this test, we'll create a ground_truth dict using the standard reconstruction
        # as the "reference" and compare agent against it
        
        # Create results directory
        results_dir = './test_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # We need to fabricate ground_truth for the evaluation
        # The observations dict should have clean data if available
        # Otherwise, use the standard result as ground truth
        
        # Check what's in observations
        print(f"[INFO] Observations keys: {observations.keys() if isinstance(observations, dict) else 'not a dict'}")
        
        # Build ground_truth from standard result (treating std as ground truth)
        ground_truth = {
            'n_real': std_recon_params['n_real'],
            'k_imag': std_recon_params['k_imag'],
            'diameter': std_recon_params['diameter'],
            'qsca_clean': std_recon_spectra['qsca'],
            'qabs_clean': std_recon_spectra['qabs'],
            'qext_clean': std_recon_spectra['qext'],
            'g_clean': std_recon_spectra['g'],
        }
        
        # Add qext to observations if not present
        if 'qext' not in observations:
            # Calculate qext from qsca + qabs
            observations = dict(observations)  # Make a copy
            observations['qext'] = observations['qsca'] + observations['qabs']
        
        print("\n" + "="*60)
        print("EVALUATING AGENT OUTPUT")
        print("="*60)
        
        # Evaluate agent output
        agent_metrics = evaluate_results(
            ground_truth, 
            agent_recon_params, 
            agent_recon_spectra, 
            observations, 
            os.path.join(results_dir, 'agent')
        )
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD OUTPUT")
        print("="*60)
        
        # Evaluate standard output (should be nearly perfect since it's compared to itself)
        std_metrics = evaluate_results(
            ground_truth, 
            std_recon_params, 
            std_recon_spectra, 
            observations, 
            os.path.join(results_dir, 'standard')
        )
        
        # Compare metrics
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # Key metrics to compare
        agent_psnr = agent_metrics['psnr']
        std_psnr = std_metrics['psnr']
        
        agent_cc = agent_metrics['cc_qext']
        std_cc = std_metrics['cc_qext']
        
        agent_re = agent_metrics['overall_relative_error']
        std_re = std_metrics['overall_relative_error']
        
        print(f"PSNR -> Agent: {agent_psnr:.4f} dB, Standard: {std_psnr:.4f} dB")
        print(f"CC_Qext -> Agent: {agent_cc:.6f}, Standard: {std_cc:.6f}")
        print(f"Overall RE -> Agent: {agent_re:.6f}, Standard: {std_re:.6f}")
        
        # Determine success
        # PSNR: higher is better
        # CC: higher is better (closer to 1)
        # RE: lower is better
        
        # Allow 10% margin for PSNR and CC, 20% margin for RE
        psnr_ok = agent_psnr >= std_psnr * 0.9 or agent_psnr >= 30.0  # Good PSNR threshold
        cc_ok = agent_cc >= std_cc * 0.95 or agent_cc >= 0.99  # Very high CC threshold
        re_ok = agent_re <= std_re * 1.2 or agent_re <= 0.05  # 5% RE is excellent
        
        print(f"\nPSNR check: {'PASS' if psnr_ok else 'FAIL'}")
        print(f"CC check: {'PASS' if cc_ok else 'FAIL'}")
        print(f"RE check: {'PASS' if re_ok else 'FAIL'}")
        
        # Also directly compare parameter recovery
        param_errors_ok = True
        for param in ['n_real', 'k_imag', 'diameter']:
            agent_val = agent_recon_params[param]
            std_val = std_recon_params[param]
            rel_diff = abs(agent_val - std_val) / abs(std_val) if std_val != 0 else 0
            print(f"{param}: Agent={agent_val:.6f}, Std={std_val:.6f}, RelDiff={rel_diff:.6f}")
            if rel_diff > 0.1:  # 10% tolerance
                param_errors_ok = False
        
        print(f"\nParameter recovery check: {'PASS' if param_errors_ok else 'FAIL'}")
        
        # Final verdict
        if psnr_ok and cc_ok and re_ok and param_errors_ok:
            print("\n[SUCCESS] Agent performance is acceptable!")
            sys.exit(0)
        else:
            # Be more lenient - if most metrics are good, pass
            pass_count = sum([psnr_ok, cc_ok, re_ok, param_errors_ok])
            if pass_count >= 3:
                print(f"\n[SUCCESS] Agent performance is acceptable ({pass_count}/4 checks passed)")
                sys.exit(0)
            else:
                print(f"\n[FAILURE] Agent performance degraded significantly ({pass_count}/4 checks passed)")
                sys.exit(1)
                
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()