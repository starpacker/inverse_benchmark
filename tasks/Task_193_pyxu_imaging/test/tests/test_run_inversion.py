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

# Set up repo path if exists
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

# Inject the referee function (evaluate_results) verbatim from Reference B
def evaluate_results(x_true, x_reconstructed, y_observed, config, results_dir):
    """
    Compute metrics, visualize, and save results.
    
    Parameters:
    -----------
    x_true : np.ndarray
        Ground truth image (2D)
    x_reconstructed : np.ndarray
        Reconstructed image (2D)
    y_observed : np.ndarray
        Observed data (1D flattened)
    config : dict
        Configuration parameters (lambda_tv, blur_sigma, noise_level, max_iter, img_size)
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, SSIM, RMSE, correlation coefficient
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute PSNR
    def compute_psnr(ref, test, data_range=None):
        ref = ref.astype(np.float64).ravel()
        test = test.astype(np.float64).ravel()
        if data_range is None:
            data_range = ref.max() - ref.min()
        if data_range < 1e-10:
            data_range = 1.0
        mse = np.mean((ref - test) ** 2)
        if mse < 1e-30:
            return 100.0
        return 10 * np.log10(data_range ** 2 / mse)
    
    # Compute SSIM
    def compute_ssim(ref, test):
        from skimage.metrics import structural_similarity as ssim
        r = ref.squeeze()
        t = test.squeeze()
        data_range = r.max() - r.min()
        if data_range < 1e-10:
            data_range = 1.0
        return float(ssim(r, t, data_range=data_range))
    
    # Compute RMSE
    def compute_rmse(ref, test):
        return float(np.sqrt(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)))
    
    # Compute Correlation Coefficient
    def compute_correlation(ref, test):
        r = ref.flatten().astype(np.float64)
        t = test.flatten().astype(np.float64)
        r_c = r - r.mean()
        t_c = t - t.mean()
        num = np.sum(r_c * t_c)
        den = np.sqrt(np.sum(r_c**2) * np.sum(t_c**2))
        if den < 1e-30:
            return 0.0
        return float(num / den)
    
    # Compute observation PSNR
    obs_psnr = compute_psnr(x_true.ravel(), y_observed)
    print(f"  Observation PSNR: {obs_psnr:.2f} dB (degraded)")
    
    # Compute metrics
    metrics = {
        "psnr": float(compute_psnr(x_true, x_reconstructed)),
        "ssim": float(compute_ssim(x_true, x_reconstructed)),
        "rmse": float(compute_rmse(x_true, x_reconstructed)),
        "cc": float(compute_correlation(x_true, x_reconstructed)),
        "observation_psnr": float(obs_psnr),
        "solver": "Condat-Vu (primal-dual splitting)",
        "regularizer": "Total Variation (anisotropic, L1 of gradient)",
        "lambda_tv": config["lambda_tv"],
        "blur_sigma": config["blur_sigma"],
        "noise_level": config["noise_level"],
        "max_iterations": config["max_iter"],
        "image_size": config["img_size"],
    }
    
    print(f"  PSNR = {metrics['psnr']:.4f} dB")
    print(f"  SSIM = {metrics['ssim']:.6f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CC   = {metrics['cc']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics -> {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), x_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), x_reconstructed)
    np.save("gt_output.npy", x_true)
    np.save("recon_output.npy", x_reconstructed)
    print(f"  Arrays saved")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    gt = x_true.squeeze()
    obs = y_observed.reshape(x_true.shape).squeeze()
    recon = x_reconstructed.squeeze()
    error = np.abs(gt - recon)

    vmin, vmax = 0, 1

    im0 = axes[0, 0].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('(a) Ground Truth', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(obs, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'(b) Observation (blur σ={config["blur_sigma"]}, noise={config["noise_level"]})',
                         fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'(c) Reconstruction (PSNR={metrics["psnr"]:.2f} dB)',
                         fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 0.01))
    axes[1, 0].set_title(f'(d) |Error| (RMSE={metrics["rmse"]:.4f})',
                         fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    mid_row = gt.shape[0] // 2
    axes[1, 1].plot(gt[mid_row, :], 'b-', label='GT', linewidth=2)
    axes[1, 1].plot(obs[mid_row, :], 'g--', label='Observed', linewidth=1, alpha=0.5)
    axes[1, 1].plot(recon[mid_row, :], 'r-', label='Recon', linewidth=1.5)
    axes[1, 1].set_title(f'(e) Profile at row {mid_row}', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    try:
        import pyxu
        pyxu_ver = pyxu.__version__
    except:
        pyxu_ver = "unknown"
    metrics_text = (
        f"Reconstruction Metrics\n"
        f"{'='*30}\n\n"
        f"PSNR:  {metrics['psnr']:.2f} dB\n"
        f"SSIM:  {metrics['ssim']:.4f}\n"
        f"RMSE:  {metrics['rmse']:.6f}\n"
        f"CC:    {metrics['cc']:.4f}\n\n"
        f"{'='*30}\n"
        f"Solver: Condat-Vu (primal-dual)\n"
        f"Library: Pyxu {pyxu_ver}\n"
        f"lambda_TV: {config['lambda_tv']}\n"
        f"Blur sigma: {config['blur_sigma']}\n"
        f"Noise: {config['noise_level']}\n"
        f"Image: {config['img_size']}x{config['img_size']}\n"
        f"Max iter: {config['max_iter']}"
    )
    axes[1, 2].text(0.1, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Pyxu Image Deconvolution: TV-Regularized Proximal Algorithm',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization -> {vis_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyxu_imaging_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    # Check which execution pattern we have
    is_chained = len(inner_files) > 0
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent's function
        print("\nRunning agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print(f"Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
        
        if is_chained:
            # Chained execution - agent_output is a callable
            inner_path = inner_files[0]
            print(f"\nLoading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print("Running chained execution...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        print(f"\nFinal result shape: {final_result.shape if hasattr(final_result, 'shape') else type(final_result)}")
        print(f"Standard result shape: {std_result.shape if hasattr(std_result, 'shape') else type(std_result)}")
        
        # We need ground truth and config to evaluate
        # Extract from args based on function signature:
        # run_inversion(y_observed, img_shape, kernel, lambda_tv, max_iter)
        y_observed = args[0] if len(args) > 0 else kwargs.get('y_observed')
        img_shape = args[1] if len(args) > 1 else kwargs.get('img_shape')
        kernel = args[2] if len(args) > 2 else kwargs.get('kernel')
        lambda_tv = args[3] if len(args) > 3 else kwargs.get('lambda_tv')
        max_iter = args[4] if len(args) > 4 else kwargs.get('max_iter')
        
        print(f"\nExtracted parameters:")
        print(f"  y_observed shape: {y_observed.shape if hasattr(y_observed, 'shape') else type(y_observed)}")
        print(f"  img_shape: {img_shape}")
        print(f"  kernel shape: {kernel.shape if hasattr(kernel, 'shape') else type(kernel)}")
        print(f"  lambda_tv: {lambda_tv}")
        print(f"  max_iter: {max_iter}")
        
        # For evaluation, we need x_true (ground truth)
        # Since this is a deconvolution problem, the standard output IS the reconstruction
        # We'll use std_result as a proxy for ground truth comparison
        # Create a synthetic ground truth for evaluation metrics comparison
        
        # Build config
        config = {
            "lambda_tv": lambda_tv if lambda_tv is not None else 0.01,
            "blur_sigma": 2.0,  # Default assumption
            "noise_level": 0.01,  # Default assumption
            "max_iter": max_iter if max_iter is not None else 100,
            "img_size": img_shape[0] if img_shape is not None else 64,
        }
        
        # Create results directories
        results_dir_agent = "./results_agent"
        results_dir_std = "./results_std"
        
        # For proper evaluation, we need x_true
        # Since std_result is the expected reconstruction, we'll compare agent vs std
        # Use std_result as "ground truth" for relative comparison
        x_true_proxy = std_result
        
        print("\n" + "="*60)
        print("Evaluating AGENT output:")
        print("="*60)
        metrics_agent = evaluate_results(
            x_true=x_true_proxy,
            x_reconstructed=final_result,
            y_observed=y_observed,
            config=config,
            results_dir=results_dir_agent
        )
        
        print("\n" + "="*60)
        print("Evaluating STANDARD output (self-comparison):")
        print("="*60)
        metrics_std = evaluate_results(
            x_true=x_true_proxy,
            x_reconstructed=std_result,
            y_observed=y_observed,
            config=config,
            results_dir=results_dir_std
        )
        
        # Extract primary metrics for comparison
        psnr_agent = metrics_agent['psnr']
        psnr_std = metrics_std['psnr']
        ssim_agent = metrics_agent['ssim']
        ssim_std = metrics_std['ssim']
        rmse_agent = metrics_agent['rmse']
        rmse_std = metrics_std['rmse']
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"PSNR  -> Agent: {psnr_agent:.4f} dB, Standard: {psnr_std:.4f} dB")
        print(f"SSIM  -> Agent: {ssim_agent:.6f}, Standard: {ssim_std:.6f}")
        print(f"RMSE  -> Agent: {rmse_agent:.6f}, Standard: {rmse_std:.6f}")
        
        # Direct array comparison
        if final_result.shape == std_result.shape:
            diff = np.abs(final_result - std_result)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"\nDirect comparison:")
            print(f"  Max absolute difference: {max_diff:.8f}")
            print(f"  Mean absolute difference: {mean_diff:.8f}")
        
        # Verification logic
        # PSNR: Higher is better
        # SSIM: Higher is better (max 1.0)
        # RMSE: Lower is better
        
        # Allow 10% margin for PSNR/SSIM degradation
        # For self-comparison with std_result as ground truth:
        # - Agent PSNR should be close to infinite (or very high) if matching
        # - Actually, if agent output equals std output, PSNR would be infinite
        
        # More practical check: direct similarity
        tolerance = 0.1  # 10% tolerance
        
        # Check if results are nearly identical
        if final_result.shape == std_result.shape:
            correlation = np.corrcoef(final_result.flatten(), std_result.flatten())[0, 1]
            print(f"\nCorrelation between agent and standard: {correlation:.6f}")
            
            # Check RMSE between agent and standard directly
            direct_rmse = np.sqrt(np.mean((final_result - std_result) ** 2))
            print(f"Direct RMSE (agent vs standard): {direct_rmse:.8f}")
            
            # Success criteria:
            # 1. High correlation (> 0.95)
            # 2. Low direct RMSE (< 0.1)
            # 3. Or PSNR is very high (> 30 dB)
            
            success = False
            
            if correlation > 0.95:
                print("\n✓ High correlation achieved!")
                success = True
            
            if direct_rmse < 0.1:
                print("✓ Low direct RMSE achieved!")
                success = True
            
            if psnr_agent > 30:
                print("✓ High PSNR achieved!")
                success = True
            
            # Also check if shapes match and values are reasonable
            if np.allclose(final_result, std_result, rtol=0.1, atol=0.05):
                print("✓ Arrays are close (within tolerance)!")
                success = True
            
            if success:
                print("\n" + "="*60)
                print("TEST PASSED: Agent performance is acceptable")
                print("="*60)
                sys.exit(0)
            else:
                print("\n" + "="*60)
                print("TEST FAILED: Agent performance degraded significantly")
                print("="*60)
                sys.exit(1)
        else:
            print(f"\nERROR: Shape mismatch!")
            print(f"  Agent shape: {final_result.shape}")
            print(f"  Standard shape: {std_result.shape}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()