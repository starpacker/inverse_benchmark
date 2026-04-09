import sys
import os
import dill
import numpy as np
import traceback
import time
from pathlib import Path

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluation
import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Inject the referee (evaluation logic) from Reference B
def visualize(gt, degraded, recon, metrics, save_path):
    """4-panel figure: GT | Degraded | Reconstruction | Error map."""
    error = np.abs(gt - recon)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ['Ground Truth', 'Degraded (Blur+Noise)',
              f'DPS Reconstruction\nPSNR={metrics["psnr_db"]:.2f} dB  '
              f'SSIM={metrics["ssim"]:.4f}',
              'Absolute Error']

    for ax, img, title in zip(axes, [gt, degraded, recon, error], titles):
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.colorbar(axes[3].images[0], ax=axes[3], fraction=0.046)

    plt.suptitle('Diffusion Posterior Sampling (DPS) — Image Deblurring',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Saved visualization to {save_path}")

def evaluate_results(gt_np: np.ndarray, degraded_np: np.ndarray,
                     recon_np: np.ndarray, config: dict,
                     results_dir: Path, elapsed_time: float):
    """
    Evaluate and save results.
    
    Args:
        gt_np: Ground truth image as numpy array
        degraded_np: Degraded image as numpy array
        recon_np: Reconstructed image as numpy array
        config: Configuration dictionary
        results_dir: Directory to save results
        elapsed_time: Total elapsed time
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n[6/6] Evaluating reconstruction ...")
    
    # Compute metrics for reconstruction
    gt_f = gt_np.astype(np.float64)
    recon_f = recon_np.astype(np.float64)
    psnr = compute_psnr(gt_f, recon_f, data_range=1.0)
    ssim = compute_ssim(gt_f, recon_f, data_range=1.0)
    rmse = np.sqrt(np.mean((gt_f - recon_f) ** 2))
    
    metrics = {'psnr_db': float(psnr), 'ssim': float(ssim), 'rmse': float(rmse)}
    
    print(f"  PSNR  = {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM  = {metrics['ssim']:.4f}")
    print(f"  RMSE  = {metrics['rmse']:.6f}")
    
    # Compute metrics for degraded image
    degraded_f = degraded_np.astype(np.float64)
    deg_psnr = compute_psnr(gt_f, degraded_f, data_range=1.0)
    deg_ssim = compute_ssim(gt_f, degraded_f, data_range=1.0)
    
    deg_metrics = {'psnr_db': float(deg_psnr), 'ssim': float(deg_ssim)}
    
    # Save metrics JSON
    metrics_out = {
        'psnr_db': metrics['psnr_db'],
        'ssim': metrics['ssim'],
        'rmse': metrics['rmse'],
        'degraded_psnr_db': deg_metrics['psnr_db'],
        'degraded_ssim': deg_metrics['ssim'],
        'method': 'Diffusion Posterior Sampling (DPS)',
        'inverse_problem': 'Gaussian deblurring',
        'image_size': config['img_size'],
        'diffusion_steps': config['num_timesteps'],
        'blur_sigma': config['blur_sigma'],
        'noise_std': config['noise_std'],
        'elapsed_seconds': elapsed_time,
    }
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")
    
    # Save numpy arrays
    np.save(results_dir / 'ground_truth.npy', gt_np)
    np.save(results_dir / 'reconstruction.npy', recon_np)
    np.save(results_dir / 'degraded.npy', degraded_np)
    print(f"  Saved .npy arrays to {results_dir}")
    
    # Visualization
    vis_path = results_dir / 'reconstruction_result.png'
    visualize(gt_np, degraded_np, recon_np, metrics, vis_path)
    
    print(f"\n{'='*60}")
    print(f" DPS Deblurring Complete")
    print(f" PSNR = {metrics['psnr_db']:.2f} dB   SSIM = {metrics['ssim']:.4f}")
    print(f" Elapsed: {elapsed_time:.1f}s")
    print(f"{'='*60}\n")
    
    return metrics


def simple_evaluate(gt_np: np.ndarray, recon_np: np.ndarray):
    """
    Simple evaluation function that computes PSNR and SSIM.
    Returns a dictionary with metrics.
    """
    gt_f = gt_np.astype(np.float64)
    recon_f = recon_np.astype(np.float64)
    
    # Ensure arrays are 2D for ssim
    if gt_f.ndim > 2:
        gt_f = gt_f.squeeze()
    if recon_f.ndim > 2:
        recon_f = recon_f.squeeze()
    
    psnr = compute_psnr(gt_f, recon_f, data_range=1.0)
    ssim = compute_ssim(gt_f, recon_f, data_range=1.0)
    rmse = np.sqrt(np.mean((gt_f - recon_f) ** 2))
    
    return {'psnr_db': float(psnr), 'ssim': float(ssim), 'rmse': float(rmse)}


def main():
    # Data paths
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        if 'parent_function' in path or 'parent_' in path:
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
        print(f"\nLoading outer data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys() if kwargs else 'None'}")
        
        # Execute the agent's run_inversion
        print("\n" + "="*60)
        print("Running agent's run_inversion...")
        print("="*60)
        
        start_time = time.time()
        agent_output = run_inversion(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        print(f"\nAgent execution completed in {elapsed_time:.1f}s")
        print(f"Agent output type: {type(agent_output)}")
        print(f"Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else 'N/A'}")
        
        # Check if there are inner data files (chained execution)
        if inner_data_paths:
            # Chained execution pattern
            print("\nDetected chained execution pattern")
            inner_data_path = inner_data_paths[0]
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            # Execute the returned operator
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print(f"\nFinal result type: {type(final_result)}")
        print(f"Standard result type: {type(std_result)}")
        
        # Convert results to numpy if needed
        if hasattr(final_result, 'cpu'):
            final_result_np = final_result.cpu().numpy()
        elif hasattr(final_result, 'numpy'):
            final_result_np = final_result.numpy()
        else:
            final_result_np = np.array(final_result)
        
        if hasattr(std_result, 'cpu'):
            std_result_np = std_result.cpu().numpy()
        elif hasattr(std_result, 'numpy'):
            std_result_np = std_result.numpy()
        else:
            std_result_np = np.array(std_result)
        
        # Squeeze if needed
        final_result_np = final_result_np.squeeze()
        std_result_np = std_result_np.squeeze()
        
        print(f"\nFinal result shape: {final_result_np.shape}")
        print(f"Standard result shape: {std_result_np.shape}")
        
        # Get ground truth from input args
        # Based on the function signature: run_inversion(y_obs, gt_tensor, blur_kernel, schedule, ...)
        # gt_tensor is the second argument
        gt_tensor = args[1] if len(args) > 1 else kwargs.get('gt_tensor')
        
        if gt_tensor is not None:
            if hasattr(gt_tensor, 'cpu'):
                gt_np = gt_tensor.cpu().numpy().squeeze()
            elif hasattr(gt_tensor, 'numpy'):
                gt_np = gt_tensor.numpy().squeeze()
            else:
                gt_np = np.array(gt_tensor).squeeze()
            
            print(f"Ground truth shape: {gt_np.shape}")
            
            # Evaluate agent result against ground truth
            print("\n" + "="*60)
            print("Evaluating Agent's reconstruction...")
            print("="*60)
            agent_metrics = simple_evaluate(gt_np, final_result_np)
            print(f"Agent PSNR: {agent_metrics['psnr_db']:.2f} dB")
            print(f"Agent SSIM: {agent_metrics['ssim']:.4f}")
            print(f"Agent RMSE: {agent_metrics['rmse']:.6f}")
            
            # Evaluate standard result against ground truth
            print("\n" + "="*60)
            print("Evaluating Standard reconstruction...")
            print("="*60)
            std_metrics = simple_evaluate(gt_np, std_result_np)
            print(f"Standard PSNR: {std_metrics['psnr_db']:.2f} dB")
            print(f"Standard SSIM: {std_metrics['ssim']:.4f}")
            print(f"Standard RMSE: {std_metrics['rmse']:.6f}")
            
            # Compare results
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)
            print(f"Agent PSNR:    {agent_metrics['psnr_db']:.2f} dB")
            print(f"Standard PSNR: {std_metrics['psnr_db']:.2f} dB")
            print(f"Agent SSIM:    {agent_metrics['ssim']:.4f}")
            print(f"Standard SSIM: {std_metrics['ssim']:.4f}")
            
            # Determine success
            # For PSNR and SSIM, higher is better
            # Allow 10% margin of error
            psnr_threshold = std_metrics['psnr_db'] * 0.90
            ssim_threshold = std_metrics['ssim'] * 0.90
            
            psnr_pass = agent_metrics['psnr_db'] >= psnr_threshold
            ssim_pass = agent_metrics['ssim'] >= ssim_threshold
            
            print(f"\nPSNR threshold (90%): {psnr_threshold:.2f} dB - {'PASS' if psnr_pass else 'FAIL'}")
            print(f"SSIM threshold (90%): {ssim_threshold:.4f} - {'PASS' if ssim_pass else 'FAIL'}")
            
            if psnr_pass and ssim_pass:
                print("\n✓ TEST PASSED: Agent performance is acceptable")
                sys.exit(0)
            else:
                print("\n✗ TEST FAILED: Agent performance degraded significantly")
                sys.exit(1)
        else:
            # No ground truth available, compare outputs directly
            print("\nNo ground truth available, comparing outputs directly...")
            
            # Compute similarity between agent and standard outputs
            diff = np.abs(final_result_np - std_result_np)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            print(f"Mean absolute difference: {mean_diff:.6f}")
            print(f"Max absolute difference: {max_diff:.6f}")
            
            # Allow reasonable tolerance
            if mean_diff < 0.1 and max_diff < 0.5:
                print("\n✓ TEST PASSED: Results are similar enough")
                sys.exit(0)
            else:
                print("\n✗ TEST FAILED: Results differ significantly")
                sys.exit(1)
                
    except Exception as e:
        print(f"\nERROR during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()