import sys
import os
import dill
import numpy as np
import traceback
import json

# Import matplotlib with Agg backend before other imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from skimage.metrics import structural_similarity as ssim

# ============================================================================
# REFEREE FUNCTION (Injected from Reference B)
# ============================================================================

def evaluate_results(ground_truth, reconstruction, measurements, results_dir):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters:
        ground_truth: tuple (tx_gt, ty_gt) ground truth traction fields in Pa
        reconstruction: tuple (tx_rec, ty_rec) reconstructed traction fields in Pa
        measurements: tuple (u_meas, v_meas) measured displacement fields in pixels
        results_dir: str, directory to save results
    
    Returns:
        metrics: dict containing all computed metrics
    """
    tx_gt, ty_gt = ground_truth
    tx_rec, ty_rec = reconstruction
    u_meas, v_meas = measurements
    
    # Helper functions for metrics
    def compute_psnr(ref, test, data_range=None):
        """Compute PSNR (dB) between reference and test arrays."""
        if data_range is None:
            data_range = ref.max() - ref.min()
        if data_range == 0:
            return float('inf') if np.allclose(ref, test) else 0.0
        mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)

    def compute_ssim(ref, test):
        """Compute SSIM for 2D fields."""
        data_range = ref.max() - ref.min()
        if data_range == 0:
            data_range = 1.0
        return ssim(ref, test, data_range=data_range)

    def compute_rmse(ref, test):
        """Compute RMSE."""
        return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))

    def compute_relative_error(ref, test):
        """Compute relative error (RE) = ||ref - test||_2 / ||ref||_2."""
        ref_norm = np.linalg.norm(ref.ravel())
        if ref_norm == 0:
            return float('inf')
        return np.linalg.norm((ref - test).ravel()) / ref_norm

    def compute_correlation_coefficient(ref, test):
        """Compute Pearson correlation coefficient."""
        ref_flat = ref.ravel()
        test_flat = test.ravel()
        if np.std(ref_flat) == 0 or np.std(test_flat) == 0:
            return 0.0
        return float(np.corrcoef(ref_flat, test_flat)[0, 1])

    # Compute metrics on traction magnitude
    gt_mag = np.sqrt(tx_gt**2 + ty_gt**2)
    rec_mag = np.sqrt(tx_rec**2 + ty_rec**2)

    metrics = {
        "psnr": float(compute_psnr(gt_mag, rec_mag)),
        "ssim": float(compute_ssim(gt_mag, rec_mag)),
        "rmse": float(compute_rmse(gt_mag, rec_mag)),
        "relative_error": float(compute_relative_error(gt_mag, rec_mag)),
        "correlation_coefficient": float(compute_correlation_coefficient(gt_mag, rec_mag)),
        # Also compute component-wise metrics
        "psnr_tx": float(compute_psnr(tx_gt, tx_rec)),
        "psnr_ty": float(compute_psnr(ty_gt, ty_rec)),
        "rmse_tx": float(compute_rmse(tx_gt, tx_rec)),
        "rmse_ty": float(compute_rmse(ty_gt, ty_rec)),
    }

    print(f"[EVAL] PSNR (magnitude) = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] SSIM (magnitude) = {metrics['ssim']:.6f}")
    print(f"[EVAL] RMSE (magnitude) = {metrics['rmse']:.4f} Pa")
    print(f"[EVAL] Relative Error   = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Correlation Coef = {metrics['correlation_coefficient']:.6f}")

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # Visualization
    disp_mag = np.sqrt(u_meas**2 + v_meas**2)
    err_mag = np.abs(gt_mag - rec_mag)

    vmax = max(gt_mag.max(), rec_mag.max())

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 1: Scalar magnitude fields
    im0 = axes[0, 0].imshow(gt_mag, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title("Ground Truth |T| (Pa)", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(disp_mag, cmap='viridis')
    axes[0, 1].set_title("Measured |u| (pixels)", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(rec_mag, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 2].set_title("Reconstructed |T| (Pa)", fontsize=12)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[0, 3].imshow(err_mag, cmap='magma')
    axes[0, 3].set_title("Error |GT - Recon| (Pa)", fontsize=12)
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # Row 2: Quiver plots
    N = tx_gt.shape[0]
    skip = max(1, N // 16)  # subsample for readability
    y_grid, x_grid = np.mgrid[:N, :N]
    sl = (slice(None, None, skip), slice(None, None, skip))

    axes[1, 0].quiver(x_grid[sl], y_grid[sl], tx_gt[sl], ty_gt[sl],
                       gt_mag[sl], cmap='hot', scale_units='xy')
    axes[1, 0].set_title("GT Traction Vectors", fontsize=12)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()

    axes[1, 1].quiver(x_grid[sl], y_grid[sl], u_meas[sl], v_meas[sl],
                       disp_mag[sl], cmap='viridis', scale_units='xy')
    axes[1, 1].set_title("Measured Displacement Vectors", fontsize=12)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()

    axes[1, 2].quiver(x_grid[sl], y_grid[sl], tx_rec[sl], ty_rec[sl],
                       rec_mag[sl], cmap='hot', scale_units='xy')
    axes[1, 2].set_title("Reconstructed Traction Vectors", fontsize=12)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].invert_yaxis()

    err_tx = tx_gt - tx_rec
    err_ty = ty_gt - ty_rec
    axes[1, 3].quiver(x_grid[sl], y_grid[sl], err_tx[sl], err_ty[sl],
                       err_mag[sl], cmap='magma', scale_units='xy')
    axes[1, 3].set_title("Error Vectors", fontsize=12)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].invert_yaxis()

    fig.suptitle(
        f"pyTFM — Traction Force Microscopy Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"RMSE={metrics['rmse']:.2f} Pa | RE={metrics['relative_error']:.4f} | "
        f"CC={metrics['correlation_coefficient']:.4f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")

    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), np.stack([tx_rec, ty_rec]))
    np.save(os.path.join(results_dir, "ground_truth.npy"), np.stack([tx_gt, ty_gt]))
    np.save(os.path.join(results_dir, "measurements.npy"), np.stack([u_meas, v_meas]))

    return metrics

# ============================================================================
# TEST LOGIC
# ============================================================================

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyTFM_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    # Determine execution pattern
    is_chained = len(inner_files) > 0
    
    try:
        # Load primary (outer) data
        if not outer_files:
            print("[ERROR] No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"[LOAD] Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Running run_inversion with {len(args)} args and {len(kwargs)} kwargs")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Chained execution: agent_output is a callable
            inner_path = inner_files[0]
            print(f"[LOAD] Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print(f"[INFO] Running chained function...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # Validate outputs
        print(f"[INFO] Agent result type: {type(final_result)}")
        print(f"[INFO] Standard result type: {type(std_result)}")
        
        # For run_inversion, the output is (tx_recon, ty_recon)
        # The input measurements contains (u, v) displacement fields
        # We need ground_truth for evaluation
        
        # Extract measurements from inputs
        measurements = args[0] if args else kwargs.get('measurements')
        
        # The standard output serves as our reference/ground truth for comparison
        # In a real scenario, ground_truth would be separate, but here we use
        # the standard output as reference to compare agent output quality
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Evaluate agent output against standard output
        print("\n[EVAL] ========== AGENT OUTPUT EVALUATION ==========")
        agent_results_dir = os.path.join(results_dir, 'agent')
        
        # Use standard result as ground truth to evaluate agent
        metrics_agent = evaluate_results(
            ground_truth=std_result,  # (tx_gt, ty_gt) 
            reconstruction=final_result,  # (tx_rec, ty_rec)
            measurements=measurements,
            results_dir=agent_results_dir
        )
        
        # For standard (baseline), compare with itself - should be perfect
        print("\n[EVAL] ========== STANDARD OUTPUT EVALUATION ==========")
        std_results_dir = os.path.join(results_dir, 'standard')
        metrics_std = evaluate_results(
            ground_truth=std_result,
            reconstruction=std_result,
            measurements=measurements,
            results_dir=std_results_dir
        )
        
        # Extract primary metrics for comparison
        # PSNR: Higher is better (dB scale)
        # SSIM: Higher is better (0-1 scale)
        # Correlation: Higher is better (-1 to 1)
        # RMSE: Lower is better
        # Relative Error: Lower is better
        
        score_agent_psnr = metrics_agent['psnr']
        score_agent_ssim = metrics_agent['ssim']
        score_agent_corr = metrics_agent['correlation_coefficient']
        score_agent_rmse = metrics_agent['rmse']
        score_agent_re = metrics_agent['relative_error']
        
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Agent PSNR:        {score_agent_psnr:.4f} dB")
        print(f"Agent SSIM:        {score_agent_ssim:.6f}")
        print(f"Agent Correlation: {score_agent_corr:.6f}")
        print(f"Agent RMSE:        {score_agent_rmse:.4f} Pa")
        print(f"Agent Rel. Error:  {score_agent_re:.6f}")
        print("="*60)
        
        # Determine success criteria
        # Since we're comparing agent output to standard output:
        # - PSNR should be high (ideally inf for identical outputs)
        # - SSIM should be close to 1
        # - Correlation should be close to 1
        # - RMSE should be low
        # - Relative Error should be low
        
        # Set thresholds for acceptable performance
        # Allow some tolerance since implementations may differ slightly
        PSNR_THRESHOLD = 20.0  # dB - reasonable reconstruction quality
        SSIM_THRESHOLD = 0.8   # Good structural similarity
        CORR_THRESHOLD = 0.9   # High correlation
        RE_THRESHOLD = 0.5     # Max 50% relative error
        
        success = True
        failure_reasons = []
        
        if score_agent_psnr < PSNR_THRESHOLD and score_agent_psnr != float('inf'):
            failure_reasons.append(f"PSNR {score_agent_psnr:.2f} < {PSNR_THRESHOLD} dB")
            success = False
            
        if score_agent_ssim < SSIM_THRESHOLD:
            failure_reasons.append(f"SSIM {score_agent_ssim:.4f} < {SSIM_THRESHOLD}")
            success = False
            
        if score_agent_corr < CORR_THRESHOLD:
            failure_reasons.append(f"Correlation {score_agent_corr:.4f} < {CORR_THRESHOLD}")
            success = False
            
        if score_agent_re > RE_THRESHOLD and score_agent_re != float('inf'):
            failure_reasons.append(f"Relative Error {score_agent_re:.4f} > {RE_THRESHOLD}")
            success = False
        
        if success:
            print("\n[PASS] Agent performance is acceptable!")
            print(f"  - PSNR: {score_agent_psnr:.2f} dB (threshold: {PSNR_THRESHOLD})")
            print(f"  - SSIM: {score_agent_ssim:.4f} (threshold: {SSIM_THRESHOLD})")
            print(f"  - Correlation: {score_agent_corr:.4f} (threshold: {CORR_THRESHOLD})")
            sys.exit(0)
        else:
            print("\n[FAIL] Agent performance degraded!")
            for reason in failure_reasons:
                print(f"  - {reason}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()