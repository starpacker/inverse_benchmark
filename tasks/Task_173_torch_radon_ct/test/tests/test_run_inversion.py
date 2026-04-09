import sys
import os
import dill
import numpy as np
import traceback
import json
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Import the target function
from agent_run_inversion import run_inversion


# Inject the evaluate_results function (Reference B)
def evaluate_results(ground_truth, reconstructions_dict, results_dir):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR, SSIM, and RMSE for each reconstruction method,
    generates visualization, and saves metrics to files.
    
    Args:
        ground_truth: Ground truth image (2D numpy array)
        reconstructions_dict: Dictionary mapping method names to tuples of 
                             (reconstruction, elapsed_time, sinogram_noisy, 
                              theta, actual_snr, image_size, num_angles)
        results_dir: Directory to save results
        
    Returns:
        results: Dictionary of metrics for all methods
        best_method: Name of the best performing method
        best_metrics: Metrics dictionary for the best method
    """
    os.makedirs(results_dir, exist_ok=True)
    
    def compute_metrics(gt, rec):
        """Compute PSNR, SSIM, and RMSE."""
        gt_copy = gt.copy()
        rec_copy = np.clip(rec.copy(), 0, None)
        
        gt_max = gt_copy.max()
        if gt_max > 0:
            gt_norm = gt_copy / gt_max
            rec_norm = rec_copy / gt_max
            rec_norm = np.clip(rec_norm, 0, 1)
        else:
            gt_norm = gt_copy
            rec_norm = rec_copy
        
        psnr = peak_signal_noise_ratio(gt_norm, rec_norm, data_range=1.0)
        ssim = structural_similarity(gt_norm, rec_norm, data_range=1.0)
        rmse = np.sqrt(mean_squared_error(gt_norm, rec_norm))
        return psnr, ssim, rmse
    
    results = {}
    recons = {}
    
    # Extract common parameters from first entry
    first_key = list(reconstructions_dict.keys())[0]
    _, _, sinogram_noisy, theta, actual_snr, image_size, num_angles = reconstructions_dict[first_key]
    
    # Evaluate each method
    for method_name, (recon, elapsed_time, _, _, _, _, _) in reconstructions_dict.items():
        psnr, ssim, rmse = compute_metrics(ground_truth, recon)
        results[method_name] = {
            'psnr': psnr,
            'ssim': ssim,
            'rmse': rmse,
            'time': elapsed_time
        }
        recons[method_name] = recon
        print(f"    {method_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
              f"RMSE={rmse:.4f}, time={elapsed_time:.2f}s")
    
    # Find best methods
    best_method = max(results, key=lambda k: results[k]['psnr'])
    best_recon = recons[best_method]
    best_metrics = results[best_method]
    
    # Best iterative method
    iter_keys = [k for k in results if 'SIRT' in k]
    if iter_keys:
        best_iter = max(iter_keys, key=lambda k: results[k]['psnr'])
        best_iter_recon = recons[best_iter]
        best_iter_metrics = results[best_iter]
    else:
        best_iter = best_method
        best_iter_recon = best_recon
        best_iter_metrics = best_metrics
    
    # Best FBP method
    fbp_keys = [k for k in results if k.startswith('FBP')]
    if fbp_keys:
        best_fbp = max(fbp_keys, key=lambda k: results[k]['psnr'])
        best_fbp_recon = recons[best_fbp]
        best_fbp_metrics = results[best_fbp]
    else:
        best_fbp = best_method
        best_fbp_recon = best_recon
        best_fbp_metrics = best_metrics
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(ground_truth, cmap='gray', vmin=0, vmax=ground_truth.max())
    ax.set_title(f'(a) Ground Truth\n(Shepp-Logan Phantom {image_size}×{image_size})',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (b) Sinogram
    ax = axes[0, 1]
    im = ax.imshow(sinogram_noisy, cmap='hot', aspect='auto',
                   extent=[theta[0], theta[-1], sinogram_noisy.shape[0], 0])
    ax.set_title(f'(b) Noisy Sinogram\nSNR={actual_snr:.1f} dB, {num_angles} angles',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Detector position')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (c) Best FBP reconstruction
    ax = axes[1, 0]
    disp = np.clip(best_fbp_recon, 0, ground_truth.max())
    im = ax.imshow(disp, cmap='gray', vmin=0, vmax=ground_truth.max())
    ax.set_title(f'(c) {best_fbp} Reconstruction\nPSNR={best_fbp_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_fbp_metrics["ssim"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (d) Best iterative reconstruction
    ax = axes[1, 1]
    disp = np.clip(best_iter_recon, 0, ground_truth.max())
    im = ax.imshow(disp, cmap='gray', vmin=0, vmax=ground_truth.max())
    ax.set_title(f'(d) {best_iter} Reconstruction\nPSNR={best_iter_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_iter_metrics["ssim"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle('CT Reconstruction via Radon Transform Inversion\n'
                 f'Best: {best_method} — PSNR={best_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_metrics["ssim"]:.4f}',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved figure: {fig_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), ground_truth)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_recon)
    
    # Save metrics JSON
    metrics_output = {
        "task": "torch_radon_ct",
        "inverse_problem": "CT reconstruction via Radon transform inversion",
        "image_size": image_size,
        "num_angles": num_angles,
        "noise_snr_db": float(actual_snr),
        "best_method": best_method,
        "best_psnr_db": float(best_metrics['psnr']),
        "best_ssim": float(best_metrics['ssim']),
        "best_rmse": float(best_metrics['rmse']),
        "all_methods": {
            method: {
                "psnr_db": float(v['psnr']),
                "ssim": float(v['ssim']),
                "rmse": float(v['rmse']),
                "time_seconds": float(v['time'])
            } for method, v in results.items()
        }
    }
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print("    Saved: ground_truth.npy, reconstruction.npy, metrics.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ground truth:   {image_size}x{image_size} Shepp-Logan phantom")
    print(f"  Forward model:  Radon transform, {num_angles} angles, SNR={actual_snr:.1f} dB")
    print(f"  Methods compared:")
    for method, v in results.items():
        print(f"    {method:14s}  PSNR={v['psnr']:.2f} dB  SSIM={v['ssim']:.4f}  "
              f"RMSE={v['rmse']:.4f}  t={v['time']:.1f}s")
    print(f"  Best method:    {best_method}")
    print(f"  Best PSNR:      {best_metrics['psnr']:.2f} dB")
    print(f"  Best SSIM:      {best_metrics['ssim']:.4f}")
    print("=" * 60)
    
    return results, best_method, best_metrics


def compute_single_metrics(ground_truth, reconstruction):
    """
    Compute metrics for a single reconstruction against ground truth.
    """
    gt_copy = ground_truth.copy()
    rec_copy = np.clip(reconstruction.copy(), 0, None)
    
    gt_max = gt_copy.max()
    if gt_max > 0:
        gt_norm = gt_copy / gt_max
        rec_norm = rec_copy / gt_max
        rec_norm = np.clip(rec_norm, 0, 1)
    else:
        gt_norm = gt_copy
        rec_norm = rec_copy
    
    psnr = peak_signal_noise_ratio(gt_norm, rec_norm, data_range=1.0)
    ssim = structural_similarity(gt_norm, rec_norm, data_range=1.0)
    rmse = np.sqrt(mean_squared_error(gt_norm, rec_norm))
    return psnr, ssim, rmse


def main():
    data_paths = ['/data/yjh/torch_radon_ct_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'run_inversion')
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Function: {func_name}")
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent's implementation
        print("\nRunning agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if there are inner files (chained execution)
        if inner_files:
            print("\nChained execution detected - running inner function...")
            inner_path = inner_files[0]
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_inner_output = inner_data.get('output', None)
            
            # agent_output should be callable
            if callable(agent_output):
                final_agent_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_agent_result = agent_output
            
            final_std_result = std_inner_output
        else:
            # Direct execution
            final_agent_result = agent_output
            final_std_result = std_output
        
        # Extract reconstruction from results
        # run_inversion returns (reconstruction, elapsed_time)
        if isinstance(final_agent_result, tuple) and len(final_agent_result) >= 2:
            agent_reconstruction = final_agent_result[0]
            agent_time = final_agent_result[1]
        else:
            agent_reconstruction = final_agent_result
            agent_time = 0.0
        
        if isinstance(final_std_result, tuple) and len(final_std_result) >= 2:
            std_reconstruction = final_std_result[0]
            std_time = final_std_result[1]
        else:
            std_reconstruction = final_std_result
            std_time = 0.0
        
        print(f"\nAgent reconstruction shape: {agent_reconstruction.shape}")
        print(f"Standard reconstruction shape: {std_reconstruction.shape}")
        print(f"Agent time: {agent_time:.4f}s, Standard time: {std_time:.4f}s")
        
        # We need a ground truth to evaluate
        # Since we're comparing reconstructions, we can use the standard output as reference
        # or we need to extract ground truth from the context
        
        # For a fair comparison, we compute metrics comparing agent vs standard
        # The standard reconstruction serves as our "ground truth" reference
        
        # Compute direct comparison metrics
        psnr_comparison = peak_signal_noise_ratio(
            std_reconstruction, 
            agent_reconstruction, 
            data_range=std_reconstruction.max() - std_reconstruction.min()
        )
        ssim_comparison = structural_similarity(
            std_reconstruction, 
            agent_reconstruction, 
            data_range=std_reconstruction.max() - std_reconstruction.min()
        )
        rmse_comparison = np.sqrt(mean_squared_error(std_reconstruction, agent_reconstruction))
        
        print(f"\n=== Comparison Metrics (Agent vs Standard) ===")
        print(f"PSNR: {psnr_comparison:.2f} dB")
        print(f"SSIM: {ssim_comparison:.4f}")
        print(f"RMSE: {rmse_comparison:.6f}")
        
        # Additionally, check array statistics
        agent_mean = np.mean(agent_reconstruction)
        std_mean = np.mean(std_reconstruction)
        agent_std = np.std(agent_reconstruction)
        std_std_val = np.std(std_reconstruction)
        
        print(f"\nAgent stats - Mean: {agent_mean:.6f}, Std: {agent_std:.6f}")
        print(f"Standard stats - Mean: {std_mean:.6f}, Std: {std_std_val:.6f}")
        
        # Determine pass/fail
        # For reconstructions, SSIM > 0.95 and PSNR > 30 dB indicates very similar results
        # We allow some tolerance since numerical differences can occur
        
        ssim_threshold = 0.90  # Allow 10% tolerance
        psnr_threshold = 25.0  # dB
        
        passed = True
        reasons = []
        
        if ssim_comparison < ssim_threshold:
            passed = False
            reasons.append(f"SSIM {ssim_comparison:.4f} < threshold {ssim_threshold}")
        
        if psnr_comparison < psnr_threshold:
            passed = False
            reasons.append(f"PSNR {psnr_comparison:.2f} < threshold {psnr_threshold}")
        
        # Also check for NaN/Inf
        if np.any(np.isnan(agent_reconstruction)):
            passed = False
            reasons.append("Agent output contains NaN values")
        
        if np.any(np.isinf(agent_reconstruction)):
            passed = False
            reasons.append("Agent output contains Inf values")
        
        print(f"\n=== Test Result ===")
        if passed:
            print("PASSED: Agent implementation produces results comparable to standard.")
            print(f"Scores -> Agent SSIM: {ssim_comparison:.4f}, PSNR: {psnr_comparison:.2f} dB")
            sys.exit(0)
        else:
            print("FAILED: Agent implementation shows significant deviation from standard.")
            for reason in reasons:
                print(f"  - {reason}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()