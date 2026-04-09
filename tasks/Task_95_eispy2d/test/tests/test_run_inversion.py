import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity

# Define paths
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_95_eispy2d"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Inject the evaluate_results function (Reference B)
def evaluate_results(chi_gt, chi_rec, gx, gy, y_noisy):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters
    ----------
    chi_gt : ndarray
        Ground truth dielectric contrast (n_grid, n_grid)
    chi_rec : ndarray
        Reconstructed dielectric contrast (n_grid, n_grid)
    gx, gy : ndarray
        Grid coordinate vectors
    y_noisy : ndarray
        Noisy scattered field measurements
        
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, SSIM, RMSE values
    """
    # Compute PSNR
    peak = np.max(np.abs(chi_gt))
    if peak == 0:
        psnr_val = 0.0
    else:
        mse = np.mean((chi_gt - chi_rec) ** 2)
        if mse < 1e-30:
            psnr_val = 100.0
        else:
            psnr_val = 10.0 * np.log10(peak ** 2 / mse)
    
    # Compute SSIM
    data_range = max(chi_gt.max() - chi_gt.min(), chi_rec.max() - chi_rec.min(), 1e-10)
    ssim_val = structural_similarity(chi_gt, chi_rec, data_range=data_range)
    
    # Compute RMSE
    rmse_val = float(np.sqrt(np.mean((chi_gt - chi_rec) ** 2)))
    
    metrics = {"PSNR": psnr_val, "SSIM": ssim_val, "RMSE": rmse_val}
    
    print(f"  PSNR = {psnr_val:.2f} dB")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  RMSE = {rmse_val:.6f}")
    
    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), chi_gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), chi_rec)
    np.save(os.path.join(RESULTS_DIR, "scattered_field.npy"), y_noisy)
    
    # Website assets
    np.save(os.path.join(ASSETS_DIR, "gt_output.npy"), chi_gt)
    np.save(os.path.join(ASSETS_DIR, "recon_output.npy"), chi_rec)
    
    # Metrics JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    extent = [gx[0] * 1e3, gx[-1] * 1e3, gy[0] * 1e3, gy[-1] * 1e3]  # mm
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    im0 = axes[0].imshow(chi_gt, extent=extent, origin="lower",
                         cmap="jet", vmin=0, vmax=1.1)
    axes[0].set_title("Ground Truth  χ(r)")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)
    
    im1 = axes[1].imshow(chi_rec, extent=extent, origin="lower",
                         cmap="jet", vmin=0, vmax=1.1)
    axes[1].set_title("Reconstructed  χ̂(r)")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)
    
    diff = np.abs(chi_gt - chi_rec)
    im2 = axes[2].imshow(diff, extent=extent, origin="lower", cmap="hot")
    axes[2].set_title("|Error|")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("y [mm]")
    plt.colorbar(im2, ax=axes[2], shrink=0.85)
    
    fig.suptitle(
        f"EM Inverse Scattering (Born + Tikhonov)   "
        f"PSNR={metrics['PSNR']:.2f} dB   SSIM={metrics['SSIM']:.4f}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    
    # Save plots
    vis_paths = [
        os.path.join(RESULTS_DIR, "vis_result.png"),
        os.path.join(ASSETS_DIR, "vis_result.png"),
        os.path.join(WORKING_DIR, "vis_result.png"),
    ]
    for p in vis_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/eispy2d_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze data paths to identify outer/inner pattern
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
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    try:
        # Load the outer (primary) data
        print("\nLoading outer data...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args: {len(args)} positional arguments")
        print(f"Kwargs: {list(kwargs.keys())}")
        
        # Execute the agent's run_inversion function
        print("\nExecuting agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_data_paths:
            # Chained execution pattern
            print("\nChained execution detected. Loading inner data...")
            with open(inner_data_paths[0], 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            print("Executing inner function...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Extract chi_rec from results (run_inversion returns (chi_rec, lam))
        if isinstance(final_result, tuple):
            agent_chi_rec = final_result[0]
            agent_lambda = final_result[1]
            print(f"Agent lambda: {agent_lambda}")
        else:
            agent_chi_rec = final_result
        
        if isinstance(std_result, tuple):
            std_chi_rec = std_result[0]
            std_lambda = std_result[1]
            print(f"Standard lambda: {std_lambda}")
        else:
            std_chi_rec = std_result
        
        print(f"\nAgent chi_rec shape: {agent_chi_rec.shape}")
        print(f"Standard chi_rec shape: {std_chi_rec.shape}")
        
        # Extract ground truth and other parameters from input data
        # The input 'data' dictionary should contain chi_gt
        input_data = args[0] if args else kwargs.get('data', {})
        
        # Get chi_gt from input data if available
        chi_gt = input_data.get('chi_gt', None)
        gx = input_data.get('gx', None)
        gy = input_data.get('gy', None)
        y_noisy = input_data.get('y_noisy', None)
        
        if chi_gt is None:
            # If chi_gt is not in input data, we need to use std_chi_rec as reference
            print("\nWARNING: chi_gt not found in input data. Using standard result as ground truth reference.")
            chi_gt = std_chi_rec
        
        print(f"\nGround truth chi shape: {chi_gt.shape}")
        
        # Evaluation phase
        print("\n" + "="*60)
        print("EVALUATING AGENT OUTPUT")
        print("="*60)
        metrics_agent = evaluate_results(chi_gt, agent_chi_rec, gx, gy, y_noisy)
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD OUTPUT")
        print("="*60)
        metrics_std = evaluate_results(chi_gt, std_chi_rec, gx, gy, y_noisy)
        
        # Extract primary metrics for comparison
        psnr_agent = metrics_agent['PSNR']
        psnr_std = metrics_std['PSNR']
        ssim_agent = metrics_agent['SSIM']
        ssim_std = metrics_std['SSIM']
        rmse_agent = metrics_agent['RMSE']
        rmse_std = metrics_std['RMSE']
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Scores -> Agent PSNR: {psnr_agent:.2f} dB, Standard PSNR: {psnr_std:.2f} dB")
        print(f"Scores -> Agent SSIM: {ssim_agent:.4f}, Standard SSIM: {ssim_std:.4f}")
        print(f"Scores -> Agent RMSE: {rmse_agent:.6f}, Standard RMSE: {rmse_std:.6f}")
        
        # Verification logic
        # PSNR and SSIM: Higher is better
        # RMSE: Lower is better
        # Allow 10% margin of error
        
        margin = 0.10  # 10% tolerance
        
        # For PSNR: agent should be at least 90% of standard
        psnr_threshold = psnr_std * (1 - margin) if psnr_std > 0 else psnr_std - abs(psnr_std * margin)
        psnr_pass = psnr_agent >= psnr_threshold
        
        # For SSIM: agent should be at least 90% of standard
        ssim_threshold = ssim_std * (1 - margin)
        ssim_pass = ssim_agent >= ssim_threshold
        
        # For RMSE: agent should be at most 110% of standard (lower is better)
        rmse_threshold = rmse_std * (1 + margin)
        rmse_pass = rmse_agent <= rmse_threshold
        
        print(f"\nVerification thresholds (with {margin*100}% margin):")
        print(f"  PSNR threshold: {psnr_threshold:.2f} dB (agent >= threshold) -> {'PASS' if psnr_pass else 'FAIL'}")
        print(f"  SSIM threshold: {ssim_threshold:.4f} (agent >= threshold) -> {'PASS' if ssim_pass else 'FAIL'}")
        print(f"  RMSE threshold: {rmse_threshold:.6f} (agent <= threshold) -> {'PASS' if rmse_pass else 'FAIL'}")
        
        # Overall pass/fail decision
        # We primarily focus on PSNR and SSIM as quality metrics
        overall_pass = psnr_pass and ssim_pass
        
        if overall_pass:
            print("\n" + "="*60)
            print("TEST PASSED: Agent performance is acceptable.")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("TEST FAILED: Agent performance degraded significantly.")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()