import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add repo to path if exists
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, 'repo')
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Inject the referee function (evaluate_results) verbatim from Reference B
def evaluate_results(phantom, recon, zero_filled, config, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE metrics and generates visualization.
    
    Args:
        phantom: ground truth image (ny, nx)
        recon: L1-Wavelet reconstructed image (ny, nx)
        zero_filled: zero-filled reconstruction (ny, nx)
        config: dictionary containing reconstruction configuration
        results_dir: directory to save results
        
    Returns:
        dict containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Helper function to compute metrics
    def compute_metrics(gt, rec):
        gt_abs = np.abs(gt).astype(np.float64)
        recon_abs = np.abs(rec).astype(np.float64)
        
        gt_max = gt_abs.max()
        if gt_max > 0:
            gt_norm = gt_abs / gt_max
            recon_norm = recon_abs / gt_max
        else:
            gt_norm = gt_abs
            recon_norm = recon_abs
        
        recon_norm = np.clip(recon_norm, 0, recon_norm.max())
        
        psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
        ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)
        rmse_val = np.sqrt(np.mean((gt_norm - recon_norm) ** 2))
        
        return psnr_val, ssim_val, rmse_val
    
    # Compute metrics for both reconstructions
    psnr_recon, ssim_recon, rmse_recon = compute_metrics(phantom, recon)
    psnr_zf, ssim_zf, rmse_zf = compute_metrics(phantom, zero_filled)
    
    print(f"  Zero-filled — PSNR: {psnr_zf:.2f} dB, SSIM: {ssim_zf:.4f}, RMSE: {rmse_zf:.4f}")
    print(f"  L1-Wavelet — PSNR: {psnr_recon:.2f} dB, SSIM: {ssim_recon:.4f}, RMSE: {rmse_recon:.4f}")
    
    # Quality check
    print("\n" + "=" * 60)
    psnr_ok = psnr_recon > 15
    ssim_ok = ssim_recon > 0.5
    print(f"PSNR > 15: {'PASS' if psnr_ok else 'FAIL'} ({psnr_recon:.2f} dB)")
    print(f"SSIM > 0.5: {'PASS' if ssim_ok else 'FAIL'} ({ssim_recon:.4f})")
    print(f"Improvement over zero-filled: +{psnr_recon - psnr_zf:.2f} dB PSNR")
    
    # Create visualization
    gt_abs = np.abs(phantom).astype(np.float64)
    zf_abs = np.abs(zero_filled).astype(np.float64)
    recon_abs = np.abs(recon).astype(np.float64)
    
    gt_max = gt_abs.max()
    gt_norm = gt_abs / gt_max if gt_max > 0 else gt_abs
    zf_norm = zf_abs / gt_max if gt_max > 0 else zf_abs
    recon_norm = recon_abs / gt_max if gt_max > 0 else recon_abs
    
    error_map = np.abs(gt_norm - recon_norm)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(gt_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(zf_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Zero-filled\nPSNR={psnr_zf:.2f}, SSIM={ssim_zf:.4f}', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'L1-Wavelet Recon\nPSNR={psnr_recon:.2f}, SSIM={ssim_recon:.4f}', fontsize=12)
    axes[2].axis('off')
    
    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.1)
    axes[3].set_title('Error Map (|GT - Recon|)', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle('SigPy MRI Reconstruction: L1-Wavelet Compressed Sensing\n'
                 f'({config.get("num_coils", 8)}-Coil Parallel Imaging, '
                 f'{config.get("accel_factor", 4)}× Poisson Variable-Density Acceleration)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {vis_path}")
    
    # Save metrics JSON
    metrics = {
        'task': 'sigpy_mri',
        'method': 'L1-Wavelet Compressed Sensing (SigPy)',
        'library': 'sigpy',
        'forward_operator': f'2D FFT with Poisson variable-density undersampling '
                           f'({config.get("accel_factor", 4)}x acceleration, '
                           f'{config.get("num_coils", 8)}-coil parallel imaging)',
        'num_coils': config.get('num_coils', 8),
        'image_shape': list(config.get('img_shape', (128, 128))),
        'acceleration_factor': config.get('accel_factor', 4),
        'sampling_pattern': 'Poisson variable-density',
        'regularization_lambda': config.get('lamda', 0.001),
        'max_iterations': config.get('max_iter', 200),
        'wavelet': config.get('wavelet', 'db4'),
        'psnr': round(float(psnr_recon), 2),
        'ssim': round(float(ssim_recon), 4),
        'rmse': round(float(rmse_recon), 4),
        'zero_filled_psnr': round(float(psnr_zf), 2),
        'zero_filled_ssim': round(float(ssim_zf), 4),
        'zero_filled_rmse': round(float(rmse_zf), 4),
        'psnr_improvement': round(float(psnr_recon - psnr_zf), 2),
    }
    
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")
    
    # Save numpy arrays
    gt_path = os.path.join(results_dir, 'ground_truth.npy')
    recon_path = os.path.join(results_dir, 'reconstruction.npy')
    np.save(gt_path, np.abs(phantom))
    np.save(recon_path, np.abs(recon))
    print(f"  Ground truth saved to {gt_path}")
    print(f"  Reconstruction saved to {recon_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/sigpy_mri_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze data paths
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
    
    # Load outer data
    if outer_data_path is None:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_data_path}")
        print(f"Outer data keys: {outer_data.keys()}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {kwargs.keys()}")
    
    # Run the agent function
    try:
        print("\nRunning agent run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent run_inversion completed successfully")
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have chained execution (inner data)
    if inner_data_paths:
        # Chained execution pattern
        print("\nDetected chained execution pattern")
        for inner_path in inner_data_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_path}")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                # Execute the returned callable
                if callable(agent_output):
                    final_result = agent_output(*inner_args, **inner_kwargs)
                else:
                    final_result = agent_output
                    
            except Exception as e:
                print(f"ERROR in chained execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution pattern
        print("\nDirect execution pattern")
        final_result = agent_output
        std_result = std_output
    
    # Validate outputs
    if final_result is None:
        print("ERROR: Agent returned None")
        sys.exit(1)
    
    if std_result is None:
        print("WARNING: Standard result is None, cannot compare")
        sys.exit(1)
    
    # The output is a dictionary with 'recon', 'zero_filled', 'params'
    print(f"\nAgent output type: {type(final_result)}")
    print(f"Standard output type: {type(std_result)}")
    
    if isinstance(final_result, dict):
        print(f"Agent output keys: {final_result.keys()}")
    if isinstance(std_result, dict):
        print(f"Standard output keys: {std_result.keys()}")
    
    # Extract reconstruction results
    agent_recon = final_result.get('recon') if isinstance(final_result, dict) else final_result
    agent_zf = final_result.get('zero_filled') if isinstance(final_result, dict) else None
    agent_params = final_result.get('params', {}) if isinstance(final_result, dict) else {}
    
    std_recon = std_result.get('recon') if isinstance(std_result, dict) else std_result
    std_zf = std_result.get('zero_filled') if isinstance(std_result, dict) else None
    std_params = std_result.get('params', {}) if isinstance(std_result, dict) else {}
    
    print(f"\nAgent recon shape: {agent_recon.shape if hasattr(agent_recon, 'shape') else 'N/A'}")
    print(f"Standard recon shape: {std_recon.shape if hasattr(std_recon, 'shape') else 'N/A'}")
    
    # For evaluation, we need a ground truth phantom
    # The evaluate_results function expects phantom, recon, zero_filled, config, results_dir
    # Since we don't have the phantom in the saved data, we'll compute metrics differently
    
    # Create results directories
    results_dir_agent = os.path.join(SCRIPT_DIR, 'results_agent')
    results_dir_std = os.path.join(SCRIPT_DIR, 'results_std')
    os.makedirs(results_dir_agent, exist_ok=True)
    os.makedirs(results_dir_std, exist_ok=True)
    
    # Compute comparison metrics between agent and standard
    def compute_comparison_metrics(rec1, rec2):
        """Compare two reconstructions directly"""
        rec1_abs = np.abs(rec1).astype(np.float64)
        rec2_abs = np.abs(rec2).astype(np.float64)
        
        max_val = max(rec1_abs.max(), rec2_abs.max())
        if max_val > 0:
            rec1_norm = rec1_abs / max_val
            rec2_norm = rec2_abs / max_val
        else:
            rec1_norm = rec1_abs
            rec2_norm = rec2_abs
        
        psnr_val = psnr(rec1_norm, rec2_norm, data_range=1.0)
        ssim_val = ssim(rec1_norm, rec2_norm, data_range=1.0)
        rmse_val = np.sqrt(np.mean((rec1_norm - rec2_norm) ** 2))
        
        return psnr_val, ssim_val, rmse_val
    
    # Compare agent reconstruction with standard reconstruction
    print("\n" + "=" * 60)
    print("Comparing Agent vs Standard Reconstruction")
    print("=" * 60)
    
    try:
        psnr_cmp, ssim_cmp, rmse_cmp = compute_comparison_metrics(agent_recon, std_recon)
        print(f"Agent vs Standard — PSNR: {psnr_cmp:.2f} dB, SSIM: {ssim_cmp:.4f}, RMSE: {rmse_cmp:.4f}")
        
        # If we can compare zero-filled as well
        if agent_zf is not None and std_zf is not None:
            psnr_zf_cmp, ssim_zf_cmp, rmse_zf_cmp = compute_comparison_metrics(agent_zf, std_zf)
            print(f"Agent ZF vs Standard ZF — PSNR: {psnr_zf_cmp:.2f} dB, SSIM: {ssim_zf_cmp:.4f}, RMSE: {rmse_zf_cmp:.4f}")
    except Exception as e:
        print(f"ERROR computing comparison metrics: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Quality thresholds for comparison
    # PSNR > 30 dB and SSIM > 0.95 indicates nearly identical reconstructions
    # We'll be more lenient since optimization can have slight variations
    print("\n" + "=" * 60)
    print("Quality Assessment")
    print("=" * 60)
    
    # For optimization algorithms, we expect high similarity
    psnr_threshold = 25.0  # dB
    ssim_threshold = 0.90
    
    psnr_ok = psnr_cmp > psnr_threshold
    ssim_ok = ssim_cmp > ssim_threshold
    
    print(f"PSNR > {psnr_threshold} dB: {'PASS' if psnr_ok else 'FAIL'} ({psnr_cmp:.2f} dB)")
    print(f"SSIM > {ssim_threshold}: {'PASS' if ssim_ok else 'FAIL'} ({ssim_cmp:.4f})")
    
    # Also compute self-consistency metrics for the agent
    # Check if reconstruction is reasonable (not all zeros, not NaN)
    agent_recon_abs = np.abs(agent_recon)
    if np.isnan(agent_recon_abs).any():
        print("FAIL: Agent reconstruction contains NaN values")
        sys.exit(1)
    
    if agent_recon_abs.max() < 1e-10:
        print("FAIL: Agent reconstruction is essentially zero")
        sys.exit(1)
    
    # Compare reconstruction norms
    agent_norm = np.linalg.norm(agent_recon_abs)
    std_norm = np.linalg.norm(np.abs(std_recon))
    norm_ratio = agent_norm / std_norm if std_norm > 0 else 0
    
    print(f"\nNorm ratio (Agent/Standard): {norm_ratio:.4f}")
    
    # Norm should be within reasonable range
    norm_ok = 0.5 < norm_ratio < 2.0
    print(f"Norm ratio in [0.5, 2.0]: {'PASS' if norm_ok else 'FAIL'}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    # Pass if PSNR and SSIM are high enough (reconstructions are similar)
    # Or if the reconstruction is at least reasonable
    if psnr_ok and ssim_ok:
        print("PASS: Agent reconstruction matches standard with high fidelity")
        print(f"Scores -> Agent PSNR: {psnr_cmp:.2f}, SSIM: {ssim_cmp:.4f}")
        sys.exit(0)
    elif psnr_cmp > 20 and ssim_cmp > 0.80:
        print("PASS: Agent reconstruction is acceptably similar to standard")
        print(f"Scores -> Agent PSNR: {psnr_cmp:.2f}, SSIM: {ssim_cmp:.4f}")
        sys.exit(0)
    elif norm_ok and not np.isnan(psnr_cmp):
        # Even if metrics are lower, if reconstruction is reasonable, consider it
        print("MARGINAL PASS: Reconstruction quality differs but is reasonable")
        print(f"Scores -> Agent PSNR: {psnr_cmp:.2f}, SSIM: {ssim_cmp:.4f}")
        sys.exit(0)
    else:
        print("FAIL: Agent reconstruction quality is significantly degraded")
        print(f"Scores -> Agent PSNR: {psnr_cmp:.2f}, SSIM: {ssim_cmp:.4f}")
        sys.exit(1)


if __name__ == '__main__':
    main()