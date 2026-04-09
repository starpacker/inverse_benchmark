import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json

# ============================================================================
# Inject Referee (Evaluation Logic) - Copied verbatim from Reference B
# ============================================================================

def compute_metrics_linear(gt_map_2d, recon_map_2d):
    """Compute PSNR/SSIM on normalized linear-scale maps."""
    gt_max = np.max(gt_map_2d)
    if gt_max <= 0:
        gt_max = 1.0

    gt_n = gt_map_2d / gt_max
    recon_max = np.max(recon_map_2d)
    if recon_max <= 0:
        recon_n = np.zeros_like(recon_map_2d)
    else:
        recon_n = recon_map_2d / recon_max

    # Scale recon to minimize MSE (optimal scaling)
    scale = np.sum(gt_n * recon_n) / (np.sum(recon_n**2) + 1e-30)
    recon_scaled = np.clip(recon_n * scale, 0, 1)

    psnr = peak_signal_noise_ratio(gt_n, recon_scaled, data_range=1.0)
    ssim = structural_similarity(gt_n, recon_scaled, data_range=1.0)
    return psnr, ssim

def to_db(source_map, dynamic_range=30.0):
    """Convert to dB with dynamic range."""
    mx = np.max(source_map)
    if mx <= 0:
        return np.full_like(source_map, -dynamic_range)
    n = np.maximum(source_map / mx, 10 ** (-dynamic_range / 10))
    return 10.0 * np.log10(n)

def plot_results(coords, gt_2d, bf_2d, clean_2d, nnls_2d,
                 mic_pos, metrics_dict, save_path, params):
    """4-panel plot."""
    extent = [coords[0], coords[-1], coords[0], coords[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    panels = [
        (axes[0, 0], gt_2d, 'Ground Truth Source Distribution', None),
        (axes[0, 1], bf_2d, 'Conventional Beamforming',
         f"PSNR={metrics_dict['conv']['psnr']:.2f}dB, SSIM={metrics_dict['conv']['ssim']:.4f}"),
        (axes[1, 0], clean_2d, 'CLEAN-SC Deconvolution',
         f"PSNR={metrics_dict['clean']['psnr']:.2f}dB, SSIM={metrics_dict['clean']['ssim']:.4f}"),
        (axes[1, 1], nnls_2d, 'NNLS Inversion',
         f"PSNR={metrics_dict['nnls']['psnr']:.2f}dB, SSIM={metrics_dict['nnls']['ssim']:.4f}"),
    ]

    for ax, data, title, subtitle in panels:
        db = to_db(data)
        im = ax.imshow(db, extent=extent, origin='lower', cmap='hot',
                       vmin=-30, vmax=0, aspect='equal')
        full_title = title if subtitle is None else f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=12, fontweight='bold' if subtitle is None else 'normal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.colorbar(im, ax=ax, label='Power [dB]')
        ax.scatter(mic_pos[:, 0], mic_pos[:, 1],
                   c='cyan', s=8, alpha=0.5, marker='.', zorder=5)

    freq = params['freq']
    n_mics = params['n_mics']
    wavelength = params['wavelength']
    snr_db = params['snr_db']
    
    plt.suptitle(f'Acoustic Beamforming: Source Localization\n'
                 f'({n_mics} mics, f={freq:.0f}Hz, λ={wavelength*100:.1f}cm, SNR={snr_db:.0f}dB)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {save_path}")

def evaluate_results(q_gt, reconstructions, grid_res, results_dir, coords, mic_positions, params):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR and SSIM for each method, determines best method,
    saves results to files, and generates visualization.
    
    Parameters
    ----------
    q_gt : ndarray
        Ground truth source distribution (n_grid,)
    reconstructions : dict
        Dictionary of reconstructed source maps from run_inversion
    grid_res : int
        Grid resolution
    results_dir : str
        Directory to save results
    coords : ndarray
        1D array of grid coordinates
    mic_positions : ndarray
        Microphone positions (n_mics, 3)
    params : dict
        Dictionary of parameters (freq, n_mics, wavelength, snr_db, z_focus)
        
    Returns
    -------
    dict
        Metrics dictionary containing PSNR, SSIM for each method and best method info
    """
    os.makedirs(results_dir, exist_ok=True)
    
    gt_2d = q_gt.reshape(grid_res, grid_res)
    
    # Compute metrics for each method
    conv_2d = reconstructions['conventional'].reshape(grid_res, grid_res)
    clean_2d = reconstructions['clean_sc'].reshape(grid_res, grid_res)
    nnls_2d = reconstructions['nnls'].reshape(grid_res, grid_res)
    
    psnr_bf, ssim_bf = compute_metrics_linear(gt_2d, conv_2d)
    psnr_cl, ssim_cl = compute_metrics_linear(gt_2d, clean_2d)
    psnr_nn, ssim_nn = compute_metrics_linear(gt_2d, nnls_2d)
    
    print(f"  Conventional: PSNR={psnr_bf:.2f}dB, SSIM={ssim_bf:.4f}")
    print(f"  CLEAN-SC:     PSNR={psnr_cl:.2f}dB, SSIM={ssim_cl:.4f}")
    print(f"  NNLS:         PSNR={psnr_nn:.2f}dB, SSIM={ssim_nn:.4f}")
    
    # Determine best method
    results = {
        'conventional': {'psnr': psnr_bf, 'ssim': ssim_bf, 'map': reconstructions['conventional']},
        'clean_sc': {'psnr': psnr_cl, 'ssim': ssim_cl, 'map': reconstructions['clean_sc']},
        'nnls': {'psnr': psnr_nn, 'ssim': ssim_nn, 'map': reconstructions['nnls']},
    }
    best_name = max(results, key=lambda m: results[m]['psnr'])
    best = results[best_name]
    
    print(f"\n  Best: {best_name} (PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f})")
    
    # Save arrays
    recon_2d = best['map'].reshape(grid_res, grid_res)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_2d)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_2d)
    
    # Build metrics dictionary
    metrics = {
        'psnr_db': round(best['psnr'], 2),
        'ssim': round(best['ssim'], 4),
        'best_method': best_name,
        'conventional': {'psnr_db': round(psnr_bf, 2), 'ssim': round(ssim_bf, 4)},
        'clean_sc': {'psnr_db': round(psnr_cl, 2), 'ssim': round(ssim_cl, 4)},
        'nnls': {'psnr_db': round(psnr_nn, 2), 'ssim': round(ssim_nn, 4)},
        'parameters': {
            'frequency_hz': params['freq'],
            'n_mics': params['n_mics'],
            'grid_resolution': grid_res,
            'snr_db': params['snr_db'],
            'z_focus_m': params['z_focus'],
        }
    }
    
    # Save metrics JSON
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plot
    plot_metrics = {
        'conv': {'psnr': psnr_bf, 'ssim': ssim_bf},
        'clean': {'psnr': psnr_cl, 'ssim': ssim_cl},
        'nnls': {'psnr': psnr_nn, 'ssim': ssim_nn}
    }
    
    plot_results(
        coords, gt_2d, conv_2d, clean_2d, nnls_2d,
        mic_positions, plot_metrics,
        os.path.join(results_dir, 'reconstruction_result.png'),
        params
    )
    
    return metrics

# ============================================================================
# Simplified evaluation for comparing agent vs standard outputs directly
# ============================================================================

def simple_evaluate_reconstruction(reconstructions, grid_res):
    """
    Simple evaluation that computes aggregate metrics from reconstructions.
    Returns best PSNR using self-consistency check.
    """
    # Since we don't have ground truth in the standard data for run_inversion,
    # we'll compare the outputs directly using aggregate metrics
    conv = reconstructions.get('conventional', np.array([]))
    clean = reconstructions.get('clean_sc', np.array([]))
    nnls = reconstructions.get('nnls', np.array([]))
    
    # Compute some aggregate statistics as a proxy for quality
    metrics = {}
    
    for name, arr in [('conventional', conv), ('clean_sc', clean), ('nnls', nnls)]:
        if len(arr) > 0:
            metrics[name] = {
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'sum': float(np.sum(arr)),
                'nonzero_ratio': float(np.sum(arr > 1e-10) / len(arr))
            }
    
    return metrics


def compare_reconstructions(agent_recon, std_recon, tolerance=0.1):
    """
    Compare two reconstruction dictionaries.
    Returns True if they are similar within tolerance.
    """
    all_close = True
    
    for key in ['conventional', 'clean_sc', 'nnls']:
        if key not in agent_recon or key not in std_recon:
            print(f"  Warning: Key '{key}' missing in one of the reconstructions")
            continue
        
        agent_arr = np.asarray(agent_recon[key])
        std_arr = np.asarray(std_recon[key])
        
        if agent_arr.shape != std_arr.shape:
            print(f"  Error: Shape mismatch for '{key}': agent={agent_arr.shape}, std={std_arr.shape}")
            all_close = False
            continue
        
        # Normalize both arrays for comparison
        agent_max = np.max(np.abs(agent_arr)) if np.max(np.abs(agent_arr)) > 0 else 1.0
        std_max = np.max(np.abs(std_arr)) if np.max(np.abs(std_arr)) > 0 else 1.0
        
        agent_norm = agent_arr / agent_max
        std_norm = std_arr / std_max
        
        # Compute relative difference
        diff = np.abs(agent_norm - std_norm)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Compute correlation
        correlation = np.corrcoef(agent_arr.ravel(), std_arr.ravel())[0, 1]
        
        print(f"  {key}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, correlation={correlation:.6f}")
        
        # Check if within tolerance
        if correlation < 0.9 or mean_diff > tolerance:
            print(f"    -> MISMATCH: correlation too low or difference too high")
            all_close = False
        else:
            print(f"    -> OK")
    
    return all_close


# ============================================================================
# Main Test Logic
# ============================================================================

def main():
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 70)
    print("Testing run_inversion")
    print("=" * 70)
    
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
        print("Error: No outer data file found")
        sys.exit(1)
    
    print(f"Outer data: {outer_data_path}")
    print(f"Inner data: {inner_data_paths}")
    
    try:
        # Load outer data
        print(f"\nLoading outer data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"  Function: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args count: {len(args)}")
        print(f"  Kwargs keys: {list(kwargs.keys())}")
        
        # Execute agent function
        print("\nExecuting agent run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("  Agent execution completed")
        
        # Check if this is a chained execution pattern
        if len(inner_data_paths) > 0 and callable(agent_output):
            print("\nDetected chained execution pattern...")
            
            # Load inner data
            inner_path = inner_data_paths[0]
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator
            print("Executing agent operator on inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
            
        else:
            # Direct execution pattern
            print("\nDirect execution pattern")
            final_result = agent_output
            std_result = std_output
        
        # Validate results
        print("\n" + "=" * 70)
        print("Comparing Results")
        print("=" * 70)
        
        if isinstance(final_result, dict) and isinstance(std_result, dict):
            # Both are dictionaries (expected for run_inversion)
            print("\nBoth outputs are dictionaries - comparing reconstruction results...")
            
            # Check keys match
            agent_keys = set(final_result.keys())
            std_keys = set(std_result.keys())
            
            print(f"  Agent keys: {agent_keys}")
            print(f"  Standard keys: {std_keys}")
            
            if agent_keys != std_keys:
                print(f"  Warning: Key mismatch")
            
            # Compare the reconstructions
            comparison_passed = compare_reconstructions(final_result, std_result)
            
            # Compute aggregate metrics for reporting
            print("\nAgent metrics:")
            agent_metrics = simple_evaluate_reconstruction(final_result, 
                kwargs.get('grid_res', int(np.sqrt(len(final_result.get('conventional', []))))))
            for k, v in agent_metrics.items():
                print(f"  {k}: max={v['max']:.6f}, mean={v['mean']:.6f}, sum={v['sum']:.6f}")
            
            print("\nStandard metrics:")
            std_metrics = simple_evaluate_reconstruction(std_result,
                kwargs.get('grid_res', int(np.sqrt(len(std_result.get('conventional', []))))))
            for k, v in std_metrics.items():
                print(f"  {k}: max={v['max']:.6f}, mean={v['mean']:.6f}, sum={v['sum']:.6f}")
            
            # Determine pass/fail based on comparison
            if comparison_passed:
                print("\n" + "=" * 70)
                print("TEST PASSED: Agent output matches standard within tolerance")
                print("=" * 70)
                sys.exit(0)
            else:
                print("\n" + "=" * 70)
                print("TEST FAILED: Agent output differs significantly from standard")
                print("=" * 70)
                sys.exit(1)
                
        else:
            # Fallback comparison for non-dict outputs
            print(f"\nUnexpected output types: agent={type(final_result)}, std={type(std_result)}")
            
            if isinstance(final_result, np.ndarray) and isinstance(std_result, np.ndarray):
                if final_result.shape == std_result.shape:
                    diff = np.abs(final_result - std_result)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    print(f"  Array comparison: max_diff={max_diff}, mean_diff={mean_diff}")
                    
                    if np.allclose(final_result, std_result, rtol=0.1, atol=1e-6):
                        print("\nTEST PASSED")
                        sys.exit(0)
                    else:
                        print("\nTEST FAILED: Arrays differ")
                        sys.exit(1)
                else:
                    print(f"  Shape mismatch: agent={final_result.shape}, std={std_result.shape}")
                    sys.exit(1)
            else:
                print("  Cannot compare outputs")
                sys.exit(1)
                
    except Exception as e:
        print(f"\nError during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()