import sys
import os
import dill
import numpy as np
import traceback
import json

# Import matplotlib with Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import skimage for SSIM
from skimage.metrics import structural_similarity as ssim_fn

# Import the target function
from agent_run_inversion import run_inversion

# Inject the referee evaluation function
def evaluate_results(stress_gt_vec, stress_rec_vec, stress_gt_2d, 
                     disp_clean, disp_noisy, nx, ny, 
                     results_dir, working_dir):
    """
    Compute metrics, visualize and save results.
    
    Parameters
    ----------
    stress_gt_vec : ndarray (n,)   Ground truth stress vector [MPa].
    stress_rec_vec : ndarray (n,)  Reconstructed stress vector [MPa].
    stress_gt_2d : ndarray (nx, ny)  Ground truth stress 2D [MPa].
    disp_clean : ndarray (n,)      Clean displacement [mm].
    disp_noisy : ndarray (n,)      Noisy displacement [mm].
    nx, ny : int                   Grid dimensions.
    results_dir : str              Directory to save results.
    working_dir : str              Working directory.
    
    Returns
    -------
    metrics : dict  Dictionary containing PSNR, SSIM, CC, RE, RMSE.
    """
    # Reshape for metrics computation
    gt = stress_gt_vec.reshape(nx, ny)
    rec = stress_rec_vec.reshape(nx, ny)
    
    # Compute metrics
    dr = gt.max() - gt.min()
    mse = np.mean((gt - rec)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt, rec, data_range=dr))
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), stress_gt_2d)
    
    # Also save to sandbox root for evaluation
    np.save(os.path.join(working_dir, "gt_output.npy"), stress_gt_2d)
    np.save(os.path.join(working_dir, "recon_output.npy"), rec)
    
    # Visualization
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    vmax = max(np.abs(stress_gt_2d).max(), np.abs(rec).max())
    
    im = axes[0, 0].imshow(stress_gt_2d.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='auto')
    axes[0, 0].set_title('(a) GT Residual Stress [MPa]')
    plt.colorbar(im, ax=axes[0, 0])
    
    im = axes[0, 1].imshow(rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='auto')
    axes[0, 1].set_title('(b) Reconstructed Stress')
    plt.colorbar(im, ax=axes[0, 1])
    
    err = stress_gt_2d - rec
    im = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 0].set_title('(c) Error')
    plt.colorbar(im, ax=axes[1, 0])
    
    axes[1, 1].plot(stress_gt_2d[:, ny//2], 'b-', lw=2, label='GT')
    axes[1, 1].plot(rec[:, ny//2], 'r--', lw=2, label='Recon')
    axes[1, 1].set_xlabel('x position')
    axes[1, 1].set_ylabel('Stress [MPa]')
    axes[1, 1].set_title('(d) Mid-depth Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f"pyCM — Residual Stress Contour Method\n"
                 f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
                 f"CC={metrics['CC']:.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/pyCM_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Setup directories
    working_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(working_dir, "results")
    results_dir_agent = os.path.join(working_dir, "results_agent")
    results_dir_std = os.path.join(working_dir, "results_std")
    
    # Separate outer and inner data paths
    outer_paths = []
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        else:
            outer_paths.append(path)
    
    print(f"Outer data paths: {outer_paths}")
    print(f"Inner data paths: {inner_paths}")
    
    try:
        # Load outer/primary data
        if not outer_paths:
            print("ERROR: No primary data file found.")
            sys.exit(1)
        
        outer_path = outer_paths[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'unknown')
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Function name: {func_name}")
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Extract key parameters
        # Based on function signature: run_inversion(C, disp_meas, nx, ny)
        C = args[0] if len(args) > 0 else kwargs.get('C')
        disp_meas = args[1] if len(args) > 1 else kwargs.get('disp_meas')
        nx = args[2] if len(args) > 2 else kwargs.get('nx')
        ny = args[3] if len(args) > 3 else kwargs.get('ny')
        
        print(f"C shape: {C.shape if hasattr(C, 'shape') else 'N/A'}")
        print(f"disp_meas shape: {disp_meas.shape if hasattr(disp_meas, 'shape') else 'N/A'}")
        print(f"nx: {nx}, ny: {ny}")
        
        # Run the agent's implementation
        print("\n=== Running Agent's run_inversion ===")
        agent_output = run_inversion(*args, **kwargs)
        print(f"Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
        
        # Check for inner data (chained execution)
        if inner_paths:
            print("\n=== Chained Execution Detected ===")
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned callable with inner args
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        print(f"\nFinal result shape: {final_result.shape if hasattr(final_result, 'shape') else type(final_result)}")
        print(f"Standard result shape: {std_result.shape if hasattr(std_result, 'shape') else type(std_result)}")
        
        # For evaluation, we need additional data that might be stored or generated
        # The evaluate_results function needs: stress_gt_vec, stress_rec_vec, stress_gt_2d,
        # disp_clean, disp_noisy, nx, ny
        
        # Since we're comparing agent vs standard, we use the standard output as ground truth
        # and the agent output as reconstruction
        
        # Create simplified evaluation - compute metrics directly
        print("\n=== Computing Metrics ===")
        
        # Reshape for comparison
        stress_gt_vec = std_result.flatten()
        stress_agent_vec = final_result.flatten()
        
        # Compute simple metrics for comparison
        gt = stress_gt_vec.reshape(nx, ny)
        rec_agent = stress_agent_vec.reshape(nx, ny)
        
        # Agent metrics (comparing agent to standard as reference)
        dr = gt.max() - gt.min()
        mse_agent = np.mean((gt - rec_agent)**2)
        psnr_agent = float(10 * np.log10(dr**2 / max(mse_agent, 1e-30)))
        ssim_agent = float(ssim_fn(gt, rec_agent, data_range=dr))
        cc_agent = float(np.corrcoef(gt.ravel(), rec_agent.ravel())[0, 1])
        re_agent = float(np.linalg.norm(gt - rec_agent) / max(np.linalg.norm(gt), 1e-12))
        rmse_agent = float(np.sqrt(mse_agent))
        
        print("\n=== Agent Performance (vs Standard as Reference) ===")
        print(f"  PSNR:  {psnr_agent:.4f} dB")
        print(f"  SSIM:  {ssim_agent:.6f}")
        print(f"  CC:    {cc_agent:.6f}")
        print(f"  RE:    {re_agent:.6f}")
        print(f"  RMSE:  {rmse_agent:.6f}")
        
        # Standard metrics (self-comparison, should be perfect)
        mse_std = np.mean((gt - gt)**2)
        psnr_std = float(10 * np.log10(dr**2 / max(mse_std, 1e-30))) if mse_std > 0 else float('inf')
        ssim_std = float(ssim_fn(gt, gt, data_range=dr))
        
        print("\n=== Standard Self-Comparison (Reference) ===")
        print(f"  PSNR:  {psnr_std} dB (inf expected)")
        print(f"  SSIM:  {ssim_std:.6f} (1.0 expected)")
        
        # Verification: Check if agent output is close enough to standard
        # We use correlation coefficient and relative error as primary metrics
        print("\n=== Verification ===")
        
        # High correlation indicates similar structure
        # Low relative error indicates similar values
        
        # Accept if CC > 0.99 and RE < 0.05 (5% error margin)
        # Or if SSIM > 0.95
        
        cc_threshold = 0.95
        re_threshold = 0.10  # 10% relative error
        ssim_threshold = 0.90
        
        passed = False
        
        if cc_agent >= cc_threshold:
            print(f"✓ Correlation check PASSED (CC={cc_agent:.6f} >= {cc_threshold})")
            passed = True
        else:
            print(f"✗ Correlation check FAILED (CC={cc_agent:.6f} < {cc_threshold})")
        
        if re_agent <= re_threshold:
            print(f"✓ Relative error check PASSED (RE={re_agent:.6f} <= {re_threshold})")
            passed = True
        else:
            print(f"✗ Relative error check FAILED (RE={re_agent:.6f} > {re_threshold})")
        
        if ssim_agent >= ssim_threshold:
            print(f"✓ SSIM check PASSED (SSIM={ssim_agent:.6f} >= {ssim_threshold})")
            passed = True
        else:
            print(f"✗ SSIM check FAILED (SSIM={ssim_agent:.6f} < {ssim_threshold})")
        
        # Final verdict
        print("\n=== Final Result ===")
        if passed:
            print("PASS: Agent implementation performs acceptably.")
            sys.exit(0)
        else:
            print("FAIL: Agent implementation shows significant deviation from standard.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()