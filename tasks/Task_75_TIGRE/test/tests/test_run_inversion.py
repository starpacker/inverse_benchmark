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
from skimage.metrics import structural_similarity as ssim_fn

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# INJECTED REFEREE CODE (from Reference B)
# ============================================================================

def compute_metrics(gt, rec):
    """
    Compute image quality metrics between ground truth and reconstruction.
    """
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}

def visualize_results(gt, sinogram, rec_fbp, rec_sirt, angles, metrics, save_path):
    """
    Create visualization of reconstruction results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(gt, cmap='gray')
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)')

    axes[0, 1].imshow(sinogram, aspect='auto', cmap='gray')
    axes[0, 1].set_title(f'Sinogram ({len(angles)} angles)')
    axes[0, 1].set_xlabel('Detector')
    axes[0, 1].set_ylabel('Angle index')

    axes[0, 2].imshow(rec_fbp / max(rec_fbp.max(), 1e-12), cmap='gray')
    axes[0, 2].set_title('FBP Reconstruction')

    axes[1, 0].imshow(rec_sirt / max(rec_sirt.max(), 1e-12), cmap='gray')
    axes[1, 0].set_title('SIRT Reconstruction')

    err = np.abs(gt / max(gt.max(), 1e-12) - rec_sirt / max(rec_sirt.max(), 1e-12))
    axes[1, 1].imshow(err, cmap='hot')
    axes[1, 1].set_title('|Error| (SIRT)')

    # Profile comparison
    mid = gt.shape[0] // 2
    axes[1, 2].plot(gt[mid, :] / max(gt[mid, :].max(), 1e-12), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_fbp[mid, :] / max(rec_fbp[mid, :].max(), 1e-12),
                     'g--', lw=1.5, label='FBP')
    axes[1, 2].plot(rec_sirt[mid, :] / max(rec_sirt[mid, :].max(), 1e-12),
                     'r--', lw=1.5, label='SIRT')
    axes[1, 2].set_title('Central Profile')
    axes[1, 2].legend()

    n_angles_sparse = len(angles)
    fig.suptitle(
        f"TIGRE — Sparse-View CT Reconstruction ({n_angles_sparse} views)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_results(phantom, rec_fbp, rec_sirt, sinogram_noisy, angles_sparse, results_dir, working_dir):
    """
    Evaluate reconstruction results, compute metrics, save outputs and visualizations.
    
    Parameters:
    -----------
    phantom : ndarray
        Ground truth phantom image
    rec_fbp : ndarray
        FBP reconstruction result
    rec_sirt : ndarray
        SIRT reconstruction result
    sinogram_noisy : ndarray
        Noisy sinogram data
    angles_sparse : ndarray
        Projection angles in degrees
    results_dir : str
        Directory to save results
    working_dir : str
        Working directory for additional outputs
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics for the best reconstruction
    best_reconstruction : ndarray
        The reconstruction with highest correlation coefficient
    method_name : str
        Name of the best method ('FBP' or 'SIRT')
    """
    # Compute metrics for both methods
    m_fbp = compute_metrics(phantom, rec_fbp)
    m_sirt = compute_metrics(phantom, rec_sirt)
    
    print("\n[EVALUATION] Metrics Comparison:")
    print(f"  FBP:  CC={m_fbp['CC']:.4f}, PSNR={m_fbp['PSNR']:.1f} dB, SSIM={m_fbp['SSIM']:.4f}")
    print(f"  SIRT: CC={m_sirt['CC']:.4f}, PSNR={m_sirt['PSNR']:.1f} dB, SSIM={m_sirt['SSIM']:.4f}")
    
    # Choose best reconstruction based on correlation coefficient
    if m_sirt['CC'] >= m_fbp['CC']:
        rec_best = rec_sirt
        metrics = m_sirt
        method = "SIRT"
    else:
        rec_best = rec_fbp
        metrics = m_fbp
        method = "FBP"
    
    print(f"\n  → Best method: {method} (higher CC)")
    
    # Print detailed metrics
    print("\n[FINAL METRICS]:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_best)
    np.save(os.path.join(results_dir, "ground_truth.npy"), phantom)
    
    # Also save to working dir for website assets
    np.save(os.path.join(working_dir, "gt_output.npy"), phantom)
    np.save(os.path.join(working_dir, "recon_output.npy"), rec_best)
    
    # Visualization
    visualize_results(phantom, sinogram_noisy, rec_fbp, rec_sirt,
                      angles_sparse, metrics,
                      os.path.join(results_dir, "reconstruction_result.png"))
    
    return metrics, rec_best, method

# ============================================================================
# TEST LOGIC
# ============================================================================

def main():
    # Data paths
    data_paths = ['/data/yjh/TIGRE_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 60)
    print("Testing run_inversion performance")
    print("=" * 60)
    
    # Analyze data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"\nOuter data: {outer_data_path}")
    print(f"Inner data: {inner_data_paths}")
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"\nLoaded outer data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"\nFunction: {outer_data.get('func_name', 'unknown')}")
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's implementation
    try:
        print("\n" + "-" * 40)
        print("Running agent's run_inversion...")
        print("-" * 40)
        agent_output = run_inversion(*args, **kwargs)
        print(f"\nAgent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Handle chained execution if inner data exists
    if inner_data_paths:
        print("\nChained execution detected - processing inner data...")
        for inner_path in inner_data_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_output = inner_data.get('output', None)
                
                if callable(agent_output):
                    agent_output = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    final_agent_result = agent_output
    final_std_result = std_output
    
    # Compute metrics for both results
    # For run_inversion, we need ground truth to compute metrics
    # The sinogram and angles are in the input args
    # args = (sinogram, angles_deg, img_size, method, n_iter)
    
    sinogram = args[0] if len(args) > 0 else kwargs.get('sinogram')
    angles_deg = args[1] if len(args) > 1 else kwargs.get('angles_deg')
    img_size = args[2] if len(args) > 2 else kwargs.get('img_size')
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Direct comparison of reconstruction quality
    # Since we don't have ground truth phantom in the test data,
    # we compare the agent's output with the standard output
    
    print("\nComputing metrics (Agent vs Standard output)...")
    
    try:
        # Treat standard output as "ground truth" for comparison
        metrics_agent = compute_metrics(final_std_result, final_agent_result)
        print(f"\nAgent vs Standard metrics:")
        for k, v in sorted(metrics_agent.items()):
            print(f"  {k:20s} = {v}")
        
        # Also compute self-consistency metrics for standard
        metrics_std = compute_metrics(final_std_result, final_std_result)
        print(f"\nStandard self-consistency metrics (should be perfect):")
        for k, v in sorted(metrics_std.items()):
            print(f"  {k:20s} = {v}")
        
    except Exception as e:
        print(f"ERROR computing metrics: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # For reconstruction tasks, CC (Correlation Coefficient) is a good metric
    # PSNR and SSIM are also important
    
    agent_cc = metrics_agent['CC']
    agent_psnr = metrics_agent['PSNR']
    agent_ssim = metrics_agent['SSIM']
    
    print(f"\nAgent Performance:")
    print(f"  CC   = {agent_cc:.6f}")
    print(f"  PSNR = {agent_psnr:.2f} dB")
    print(f"  SSIM = {agent_ssim:.6f}")
    
    # Thresholds for acceptance
    # CC should be very high (close to 1)
    # PSNR should be high (>30 dB typically means good quality)
    # SSIM should be close to 1
    
    CC_THRESHOLD = 0.95  # Agent output should correlate highly with standard
    PSNR_THRESHOLD = 30.0  # Good reconstruction quality
    SSIM_THRESHOLD = 0.90  # High structural similarity
    
    passed = True
    
    if agent_cc < CC_THRESHOLD:
        print(f"\n[FAIL] CC ({agent_cc:.4f}) is below threshold ({CC_THRESHOLD})")
        passed = False
    else:
        print(f"\n[PASS] CC ({agent_cc:.4f}) >= threshold ({CC_THRESHOLD})")
    
    if agent_psnr < PSNR_THRESHOLD:
        print(f"[WARN] PSNR ({agent_psnr:.2f} dB) is below threshold ({PSNR_THRESHOLD} dB)")
        # PSNR warning but not failure if CC is good
    else:
        print(f"[PASS] PSNR ({agent_psnr:.2f} dB) >= threshold ({PSNR_THRESHOLD} dB)")
    
    if agent_ssim < SSIM_THRESHOLD:
        print(f"[WARN] SSIM ({agent_ssim:.4f}) is below threshold ({SSIM_THRESHOLD})")
        # SSIM warning but not failure if CC is good
    else:
        print(f"[PASS] SSIM ({agent_ssim:.4f}) >= threshold ({SSIM_THRESHOLD})")
    
    # Additional check: numerical similarity
    mse = np.mean((final_agent_result - final_std_result) ** 2)
    max_val = max(np.abs(final_std_result).max(), 1e-10)
    relative_mse = mse / (max_val ** 2)
    
    print(f"\nNumerical comparison:")
    print(f"  MSE (absolute)  = {mse:.6e}")
    print(f"  MSE (relative)  = {relative_mse:.6e}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if passed:
        print("TEST PASSED: Agent's run_inversion produces acceptable results")
        print("=" * 60)
        sys.exit(0)
    else:
        print("TEST FAILED: Agent's run_inversion output differs significantly from standard")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()