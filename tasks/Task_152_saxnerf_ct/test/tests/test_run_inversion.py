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
from skimage.metrics import structural_similarity as ssim

# Inject the evaluate_results function (Referee)
def evaluate_results(phantom, tv_recon, fbp_sparse, fbp_full, config, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        phantom: Ground truth image
        tv_recon: TV-regularized reconstruction
        fbp_sparse: FBP reconstruction from sparse data
        fbp_full: FBP reconstruction from full data
        config: Configuration dictionary
        results_dir: Directory to save results
        
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    
    def compute_psnr(gt, recon, data_range=None):
        """Compute PSNR between ground truth and reconstruction."""
        if data_range is None:
            data_range = gt.max() - gt.min()
        mse = np.mean((gt - recon) ** 2)
        if mse < 1e-12:
            return 100.0
        return 10.0 * np.log10(data_range ** 2 / mse)
    
    def compute_rmse(gt, recon):
        """Compute RMSE between ground truth and reconstruction."""
        return np.sqrt(np.mean((gt - recon) ** 2))
    
    def compute_ssim_metric(gt, recon):
        """Compute SSIM between ground truth and reconstruction."""
        data_range = gt.max() - gt.min()
        return ssim(gt, recon, data_range=data_range)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Clip reconstructions to [0, 1]
    tv_recon_clipped = np.clip(tv_recon, 0, 1)
    fbp_sparse_clipped = np.clip(fbp_sparse, 0, 1)
    fbp_full_clipped = np.clip(fbp_full, 0, 1)
    
    # Compute metrics for FBP full
    psnr_fbp_full = compute_psnr(phantom, fbp_full_clipped, data_range=1.0)
    ssim_fbp_full = compute_ssim_metric(phantom, fbp_full_clipped)
    
    # Compute metrics for FBP sparse
    psnr_fbp_sparse = compute_psnr(phantom, fbp_sparse_clipped, data_range=1.0)
    ssim_fbp_sparse = compute_ssim_metric(phantom, fbp_sparse_clipped)
    
    # Compute metrics for TV reconstruction
    psnr_tv = compute_psnr(phantom, tv_recon_clipped, data_range=1.0)
    ssim_tv = compute_ssim_metric(phantom, tv_recon_clipped)
    rmse_tv = compute_rmse(phantom, tv_recon_clipped)
    
    print(f"\n  FBP full ({config['n_full_angles']} angles): PSNR={psnr_fbp_full:.2f}dB, SSIM={ssim_fbp_full:.4f}")
    print(f"  FBP sparse ({config['n_sparse_angles']} angles): PSNR={psnr_fbp_sparse:.2f}dB, SSIM={ssim_fbp_sparse:.4f}")
    print(f"\n  FISTA-TV result: PSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.4f}, RMSE={rmse_tv:.4f}")
    print(f"  Improvement over sparse FBP: "
          f"PSNR +{psnr_tv - psnr_fbp_sparse:.2f}dB, "
          f"SSIM +{ssim_tv - ssim_fbp_sparse:.4f}")
    
    # Save numpy arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), phantom)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), tv_recon_clipped)
    np.save(os.path.join(results_dir, 'fbp_sparse.npy'), fbp_sparse_clipped)
    print("  Saved .npy files")
    
    # Create metrics dictionary
    metrics = {
        "task": "saxnerf_ct",
        "description": "Sparse-view CT reconstruction using FISTA-TV",
        "phantom_size": config['size'],
        "n_full_angles": config['n_full_angles'],
        "n_sparse_angles": config['n_sparse_angles'],
        "fbp_full": {
            "psnr_db": round(psnr_fbp_full, 4),
            "ssim": round(ssim_fbp_full, 4)
        },
        "fbp_sparse": {
            "psnr_db": round(psnr_fbp_sparse, 4),
            "ssim": round(ssim_fbp_sparse, 4)
        },
        "fista_tv": {
            "psnr_db": round(psnr_tv, 4),
            "ssim": round(ssim_tv, 4),
            "rmse": round(rmse_tv, 6),
            "n_iterations": 200,
            "tv_weight": 0.008
        },
        "improvement_over_sparse_fbp": {
            "psnr_gain_db": round(psnr_tv - psnr_fbp_sparse, 4),
            "ssim_gain": round(ssim_tv - ssim_fbp_sparse, 4)
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Ground Truth
    im0 = axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Sparse FBP
    im1 = axes[0, 1].imshow(fbp_sparse_clipped, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Sparse FBP ({config["n_sparse_angles"]} angles)\nPSNR={psnr_fbp_sparse:.1f}dB, SSIM={ssim_fbp_sparse:.3f}',
                         fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # TV Reconstruction
    im2 = axes[1, 0].imshow(tv_recon_clipped, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'FISTA-TV Recon ({config["n_sparse_angles"]} angles)\nPSNR={psnr_tv:.1f}dB, SSIM={ssim_tv:.3f}',
                         fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Error map (TV recon)
    error_map = np.abs(phantom - tv_recon_clipped)
    im3 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title(f'Error Map (|GT - TV Recon|)\nRMSE={rmse_tv:.4f}',
                         fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.suptitle('Sparse-View CT Reconstruction\n(SAX-NeRF Task: sparse projections → CT volume)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ground truth: Shepp-Logan phantom {config['size']}x{config['size']}")
    print(f"  Sparse angles: {config['n_sparse_angles']} (of {config['n_full_angles']})")
    print(f"  FBP sparse:  PSNR={psnr_fbp_sparse:.2f}dB, SSIM={ssim_fbp_sparse:.4f}")
    print(f"  FISTA-TV:    PSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.4f}, RMSE={rmse_tv:.6f}")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)
    
    return metrics


def compute_psnr_simple(gt, recon):
    """Simple PSNR computation for comparison."""
    gt_clipped = np.clip(gt, 0, 1)
    recon_clipped = np.clip(recon, 0, 1)
    mse = np.mean((gt_clipped - recon_clipped) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim_simple(gt, recon):
    """Simple SSIM computation for comparison."""
    gt_clipped = np.clip(gt, 0, 1)
    recon_clipped = np.clip(recon, 0, 1)
    data_range = 1.0
    return ssim(gt_clipped, recon_clipped, data_range=data_range)


def main():
    # Data paths provided
    data_paths = ['/data/yjh/saxnerf_ct_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Found {len(outer_files)} outer file(s) and {len(inner_files)} inner file(s)")
    
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs")
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run the agent function
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution)
    if inner_files:
        print("Chained execution detected - running inner function")
        inner_path = inner_files[0]
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Agent output should be callable
        if callable(agent_output):
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            print("WARNING: Agent output is not callable for chained execution")
            final_result = agent_output
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    # The output of run_inversion is (tv_recon, fbp_sparse, iteration_info)
    print("\n" + "=" * 60)
    print("Evaluating Results")
    print("=" * 60)
    
    # Unpack results
    if isinstance(final_result, tuple) and len(final_result) >= 2:
        agent_tv_recon = final_result[0]
        agent_fbp_sparse = final_result[1]
        agent_iter_info = final_result[2] if len(final_result) > 2 else {}
    else:
        print("ERROR: Unexpected agent output format")
        sys.exit(1)
    
    if isinstance(std_result, tuple) and len(std_result) >= 2:
        std_tv_recon = std_result[0]
        std_fbp_sparse = std_result[1]
        std_iter_info = std_result[2] if len(std_result) > 2 else {}
    else:
        print("ERROR: Unexpected standard output format")
        sys.exit(1)
    
    print(f"Agent tv_recon shape: {agent_tv_recon.shape}")
    print(f"Standard tv_recon shape: {std_tv_recon.shape}")
    
    # For evaluation, we need to compare the quality of reconstructions
    # Since evaluate_results requires phantom (ground truth), we need to generate it
    # or use relative comparison
    
    # Direct comparison between agent and standard outputs
    # Compute PSNR between agent reconstruction and standard reconstruction
    psnr_diff = compute_psnr_simple(std_tv_recon, agent_tv_recon)
    ssim_diff = compute_ssim_simple(std_tv_recon, agent_tv_recon)
    
    print(f"\nComparison between Agent and Standard reconstructions:")
    print(f"  PSNR(std, agent): {psnr_diff:.2f} dB")
    print(f"  SSIM(std, agent): {ssim_diff:.4f}")
    
    # Also compute basic statistics
    agent_mean = np.mean(agent_tv_recon)
    agent_std = np.std(agent_tv_recon)
    std_mean = np.mean(std_tv_recon)
    std_std = np.std(std_tv_recon)
    
    print(f"\nAgent reconstruction stats: mean={agent_mean:.4f}, std={agent_std:.4f}")
    print(f"Standard reconstruction stats: mean={std_mean:.4f}, std={std_std:.4f}")
    
    # Check data fit history if available
    if 'data_fit_history' in agent_iter_info and 'data_fit_history' in std_iter_info:
        agent_final_fit = agent_iter_info['data_fit_history'][-1][1] if agent_iter_info['data_fit_history'] else None
        std_final_fit = std_iter_info['data_fit_history'][-1][1] if std_iter_info['data_fit_history'] else None
        
        if agent_final_fit is not None and std_final_fit is not None:
            print(f"\nFinal data fit - Agent: {agent_final_fit:.2f}, Standard: {std_final_fit:.2f}")
    
    # Determine success based on SSIM similarity
    # SSIM > 0.95 indicates very similar results
    # PSNR > 30 dB indicates high quality match
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    
    # Success criteria:
    # 1. SSIM between agent and standard should be > 0.90 (allowing for numerical differences)
    # 2. PSNR between agent and standard should be > 25 dB
    
    ssim_threshold = 0.90
    psnr_threshold = 25.0
    
    success = True
    
    if ssim_diff < ssim_threshold:
        print(f"WARNING: SSIM ({ssim_diff:.4f}) below threshold ({ssim_threshold})")
        success = False
    else:
        print(f"PASS: SSIM ({ssim_diff:.4f}) >= threshold ({ssim_threshold})")
    
    if psnr_diff < psnr_threshold:
        print(f"WARNING: PSNR ({psnr_diff:.2f} dB) below threshold ({psnr_threshold} dB)")
        success = False
    else:
        print(f"PASS: PSNR ({psnr_diff:.2f} dB) >= threshold ({psnr_threshold} dB)")
    
    # Additional check: reconstruction values should be in reasonable range
    if agent_tv_recon.min() < -0.5 or agent_tv_recon.max() > 1.5:
        print(f"WARNING: Agent reconstruction has extreme values: [{agent_tv_recon.min():.4f}, {agent_tv_recon.max():.4f}]")
    
    print("=" * 60)
    
    if success:
        print("\nTEST PASSED: Agent reconstruction matches standard within tolerance")
        sys.exit(0)
    else:
        print("\nTEST FAILED: Agent reconstruction differs significantly from standard")
        sys.exit(1)


if __name__ == "__main__":
    main()