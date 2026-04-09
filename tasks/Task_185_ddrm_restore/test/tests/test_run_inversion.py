import sys
import os
import dill
import numpy as np
import traceback
import time
import json
from pathlib import Path

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Inject the referee function (evaluate_results)
def evaluate_results(
    ground_truth,
    reconstruction,
    lr_image,
    scale_factor=4,
    elapsed_time=0.0,
    results_dir=None,
    img_size=256,
    lr_size=64,
    noise_std=0.05,
    aa_sigma=1.0,
    tv_weight_stage1=0.08,
    tv_weight_stage2=0.04,
    num_datafid_iters=20
):
    """
    Evaluate reconstruction quality, save metrics and visualizations.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground-truth high-resolution image.
    reconstruction : ndarray
        Reconstructed high-resolution image.
    lr_image : ndarray
        Low-resolution input image.
    scale_factor : int
        Upsampling factor.
    elapsed_time : float
        Total processing time in seconds.
    results_dir : Path or str or None
        Directory to save results.
    img_size : int
        High-resolution image size.
    lr_size : int
        Low-resolution image size.
    noise_std : float
        Noise standard deviation used.
    aa_sigma : float
        Anti-aliasing blur sigma used.
    tv_weight_stage1 : float
        TV weight for stage 1.
    tv_weight_stage2 : float
        TV weight for stage 2.
    num_datafid_iters : int
        Number of data-fidelity iterations.
        
    Returns
    -------
    metrics : dict
        Dictionary containing all evaluation metrics.
    """
    # Compute metrics for DDRM reconstruction
    psnr_val = compute_psnr(ground_truth, reconstruction, data_range=1.0)
    ssim_val = compute_ssim(ground_truth, reconstruction, data_range=1.0)
    rmse_val = np.sqrt(np.mean((ground_truth - reconstruction)**2))
    
    # Compute baseline metrics (bicubic upsampling)
    lr_bicubic = np.clip(zoom(lr_image, scale_factor, order=3), 0, 1)
    psnr_bic = compute_psnr(ground_truth, lr_bicubic, data_range=1.0)
    ssim_bic = compute_ssim(ground_truth, lr_bicubic, data_range=1.0)
    rmse_bic = np.sqrt(np.mean((ground_truth - lr_bicubic)**2))
    
    print(f"[Baseline] Bicubic: PSNR={psnr_bic:.2f} dB, SSIM={ssim_bic:.4f}")
    
    print(f"\n{'=' * 55}")
    print(f"  Results")
    print(f"  {'─' * 50}")
    print(f"  Baseline (Bicubic)  PSNR = {psnr_bic:.4f} dB")
    print(f"  Baseline (Bicubic)  SSIM = {ssim_bic:.4f}")
    print(f"  {'─' * 50}")
    print(f"  DDRM Restoration    PSNR = {psnr_val:.4f} dB")
    print(f"  DDRM Restoration    SSIM = {ssim_val:.4f}")
    print(f"  DDRM Restoration    RMSE = {rmse_val:.4f}")
    print(f"  {'─' * 50}")
    print(f"  Improvement         PSNR = +{psnr_val - psnr_bic:.2f} dB")
    print(f"  Improvement         SSIM = +{ssim_val - ssim_bic:.4f}")
    print(f"  Time                     = {elapsed_time:.2f} s")
    print(f"{'=' * 55}")
    
    metrics = {
        "psnr_db": round(psnr_val, 4),
        "ssim": round(ssim_val, 4),
        "rmse": round(rmse_val, 4),
        "baseline_psnr_db": round(psnr_bic, 4),
        "baseline_ssim": round(ssim_bic, 4),
        "baseline_rmse": round(rmse_bic, 4),
        "method": "DDRM_SVD_restoration",
        "task": "4x_super_resolution",
        "image_size": img_size,
        "lr_size": lr_size,
        "scale_factor": scale_factor,
        "noise_std": noise_std,
        "aa_blur_sigma": aa_sigma,
        "tv_weight_stage1": tv_weight_stage1,
        "tv_weight_stage2": tv_weight_stage2,
        "num_datafid_iters": num_datafid_iters,
        "elapsed_seconds": round(elapsed_time, 2)
    }
    
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[Save] Metrics -> {metrics_path}")
        
        # Save arrays
        gt_path = results_dir / 'ground_truth.npy'
        recon_path = results_dir / 'reconstruction.npy'
        np.save(gt_path, ground_truth)
        np.save(recon_path, reconstruction)
        print(f"[Save] Ground truth -> {gt_path}")
        print(f"[Save] Reconstruction -> {recon_path}")
        
        # Create visualization
        print("[Viz] Creating 4-panel visualization ...")
        error_map = np.abs(ground_truth - reconstruction)
        
        # Upsample LR for display (nearest-neighbor to show pixelation)
        lr_display = np.repeat(np.repeat(lr_image, scale_factor, axis=0),
                               scale_factor, axis=1)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ['Ground Truth',
                  f'Low-Res Input ({scale_factor}x)',
                  'DDRM Reconstruction',
                  'Error Map']
        images = [ground_truth, lr_display, reconstruction, error_map]
        cmaps = ['gray', 'gray', 'gray', 'hot']
        
        for ax, img, title, cmap in zip(axes, images, titles, cmaps):
            vmax = 1.0 if cmap == 'gray' else max(error_map.max(), 0.01)
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(
            f'DDRM SVD-Based Super-Resolution ({scale_factor}x)  |  '
            f'PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  '
            f'RMSE: {rmse_val:.4f}',
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        vis_path = results_dir / 'reconstruction_result.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[Viz] Saved to {vis_path}")
        
        print(f"\n[Done] All outputs saved to {results_dir}")
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/ddrm_restore_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[Info] Outer files: {outer_files}")
    print(f"[Info] Inner files: {inner_files}")
    
    try:
        # Load outer data
        if not outer_files:
            print("[Error] No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"[Load] Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[Info] Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"[Info] Args count: {len(args)}")
        print(f"[Info] Kwargs keys: {list(kwargs.keys())}")
        
        # Execute agent's run_inversion
        print("\n[Exec] Running agent's run_inversion...")
        start_time = time.time()
        agent_output = run_inversion(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[Exec] Agent execution completed in {elapsed_time:.2f}s")
        
        # Check for chained execution
        if inner_files:
            # Chained execution pattern
            inner_path = inner_files[0]
            print(f"\n[Load] Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            if callable(agent_output):
                print("[Exec] Agent output is callable, executing inner call...")
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print(f"\n[Info] Agent result shape: {final_result.shape if hasattr(final_result, 'shape') else 'N/A'}")
        print(f"[Info] Standard result shape: {std_result.shape if hasattr(std_result, 'shape') else 'N/A'}")
        
        # For evaluation, we need ground truth and lr_image
        # These should be derivable from the input args
        # y_obs is the low-resolution observation (first arg)
        # target_size is the second arg
        y_obs = args[0] if len(args) > 0 else kwargs.get('y_obs')
        target_size = args[1] if len(args) > 1 else kwargs.get('target_size', 256)
        scale_factor = kwargs.get('scale_factor', 4)
        
        # For evaluation without ground truth, we compare agent vs standard output
        # We'll use the standard output as the reference for comparison
        print("\n" + "="*60)
        print("EVALUATING AGENT OUTPUT")
        print("="*60)
        
        # Direct comparison metrics between agent and standard outputs
        if std_result is not None and final_result is not None:
            # Compute direct comparison
            mse = np.mean((final_result - std_result)**2)
            rmse = np.sqrt(mse)
            max_diff = np.max(np.abs(final_result - std_result))
            
            print(f"\n[Comparison] Agent vs Standard:")
            print(f"  MSE:      {mse:.8f}")
            print(f"  RMSE:     {rmse:.8f}")
            print(f"  Max Diff: {max_diff:.8f}")
            
            # Compute PSNR between agent and standard (treating standard as reference)
            if mse > 0:
                psnr_diff = 10 * np.log10(1.0 / mse)
            else:
                psnr_diff = float('inf')
            print(f"  PSNR:     {psnr_diff:.2f} dB")
            
            # Compute SSIM between agent and standard
            ssim_diff = compute_ssim(std_result, final_result, data_range=1.0)
            print(f"  SSIM:     {ssim_diff:.4f}")
            
            # Also compute self-consistency metrics
            # Compare agent output with bicubic upsampling baseline
            lr_bicubic = np.clip(zoom(y_obs, scale_factor, order=3), 0, 1)
            
            psnr_agent_vs_bicubic = compute_psnr(lr_bicubic, final_result, data_range=1.0)
            psnr_std_vs_bicubic = compute_psnr(lr_bicubic, std_result, data_range=1.0)
            
            print(f"\n[Quality Check vs Bicubic baseline]:")
            print(f"  Agent PSNR vs Bicubic: {psnr_agent_vs_bicubic:.2f} dB")
            print(f"  Std PSNR vs Bicubic:   {psnr_std_vs_bicubic:.2f} dB")
            
            # Determine success
            # The agent should produce results very close to the standard
            # Using SSIM > 0.95 and PSNR > 30 dB as thresholds
            success = True
            
            if ssim_diff < 0.90:
                print(f"\n[WARN] SSIM between agent and standard is below 0.90")
                success = False
            
            if psnr_diff < 25:
                print(f"\n[WARN] PSNR between agent and standard is below 25 dB")
                success = False
            
            if max_diff > 0.5:
                print(f"\n[WARN] Max pixel difference exceeds 0.5")
                success = False
            
            # Additional sanity checks on the output
            if final_result.min() < -0.1 or final_result.max() > 1.1:
                print(f"\n[WARN] Agent output has values outside [0, 1] range: [{final_result.min():.4f}, {final_result.max():.4f}]")
                success = False
            
            print("\n" + "="*60)
            if success:
                print("[RESULT] TEST PASSED - Agent performance is acceptable")
                print("="*60)
                sys.exit(0)
            else:
                print("[RESULT] TEST FAILED - Agent performance degraded significantly")
                print("="*60)
                sys.exit(1)
        else:
            print("[Error] Could not compare results - missing data")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[Error] Exception during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()