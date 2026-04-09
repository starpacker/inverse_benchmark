import sys
import os
import dill
import numpy as np
import traceback
import json

# Import target function
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

# Inject the referee function (evaluate_results)
def evaluate_results(rho_gt, rho_rec, transient, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Parameters:
        rho_gt: ndarray (nx, ny, nz) - Ground truth hidden scene
        rho_rec: ndarray (nx, ny, nz) - Reconstructed hidden scene
        transient: ndarray (nx, ny, n_time) - Transient measurements
        results_dir: str - Directory to save results
    
    Returns:
        metrics: dict containing PSNR, SSIM, CC, RE, RMSE
    """
    # Max intensity projections for comparison
    gt_mip = rho_gt.max(axis=2)
    rec_mip = rho_rec.max(axis=2)
    
    # Normalize GT to [0, 1]
    gt_n = gt_mip / max(gt_mip.max(), 1e-12)
    
    # Least-squares alignment of rec_mip to gt_mip
    rec_flat = rec_mip.ravel()
    gt_flat = gt_n.ravel()
    A_mat = np.column_stack([rec_flat, np.ones_like(rec_flat)])
    result = np.linalg.lstsq(A_mat, gt_flat, rcond=None)
    a, b = result[0]
    rec_n = np.clip(a * rec_mip + b, 0, 1)
    
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Save metrics and data
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), rho_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), rho_gt)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # GT MIP
    axes[0, 0].imshow(gt_mip, cmap='hot', origin='lower')
    axes[0, 0].set_title('GT — Max Intensity Projection (XY)')
    
    # Recon MIP
    axes[0, 1].imshow(rec_mip / max(rec_mip.max(), 1e-12), cmap='hot', origin='lower')
    axes[0, 1].set_title('Recon — MIP (XY)')
    
    # Transient slice
    mid_x = transient.shape[0] // 2
    axes[0, 2].imshow(transient[mid_x, :, :].T, aspect='auto', cmap='viridis',
                       origin='lower')
    axes[0, 2].set_title(f'Transient τ(x={mid_x}, y, t)')
    axes[0, 2].set_xlabel('y index')
    axes[0, 2].set_ylabel('Time bin')
    
    # GT depth slice
    gt_side = rho_gt.max(axis=1)
    axes[1, 0].imshow(gt_side.T, cmap='hot', origin='lower', aspect='auto')
    axes[1, 0].set_title('GT — MIP (XZ)')
    
    # Recon depth slice
    rec_side = rho_rec.max(axis=1)
    axes[1, 1].imshow(rec_side.T / max(rec_side.max(), 1e-12),
                       cmap='hot', origin='lower', aspect='auto')
    axes[1, 1].set_title('Recon — MIP (XZ)')
    
    # Error
    err = np.abs(gt_mip - rec_mip / max(rec_mip.max(), 1e-12))
    axes[1, 2].imshow(err, cmap='hot', origin='lower')
    axes[1, 2].set_title('|Error| (XY)')
    
    fig.suptitle(
        f"NLOS — Non-Line-of-Sight Reconstruction\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    data_paths = ['/data/yjh/NLOS_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Categorize data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    print(f"Loading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's implementation
    try:
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent execution completed successfully.")
    except Exception as e:
        print(f"ERROR during agent execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if chained execution is needed
    if len(inner_data_paths) > 0 and callable(agent_output):
        # Chained execution pattern
        print("Detected chained execution pattern...")
        for inner_path in inner_data_paths:
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            
            agent_final = agent_output(*inner_args, **inner_kwargs)
            std_final = inner_data.get('output', None)
    else:
        # Direct execution pattern
        agent_final = agent_output
        std_final = std_output
    
    # Extract reconstruction results
    if isinstance(agent_final, dict):
        agent_rho_rec = agent_final.get('rho_rec', None)
    else:
        agent_rho_rec = agent_final
    
    if isinstance(std_final, dict):
        std_rho_rec = std_final.get('rho_rec', None)
    else:
        std_rho_rec = std_final
    
    # Get ground truth and transient from inputs
    # Based on the function signature: run_inversion(transient, x_wall, y_wall, z_scene, dt, rho_gt=None)
    transient = args[0] if len(args) > 0 else kwargs.get('transient')
    rho_gt = kwargs.get('rho_gt', None)
    if rho_gt is None and len(args) > 5:
        rho_gt = args[5]
    
    if rho_gt is None:
        print("WARNING: No ground truth (rho_gt) provided in input data.")
        print("Creating synthetic ground truth for evaluation...")
        # Use the standard output as a proxy for ground truth
        if std_rho_rec is not None:
            rho_gt = std_rho_rec
        else:
            print("ERROR: Cannot evaluate without ground truth!")
            sys.exit(1)
    
    # Create separate results directories for agent and standard
    agent_results_dir = os.path.join(RESULTS_DIR, "agent")
    std_results_dir = os.path.join(RESULTS_DIR, "standard")
    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(std_results_dir, exist_ok=True)
    
    # Evaluate both results
    print("\nEvaluating agent's reconstruction...")
    try:
        metrics_agent = evaluate_results(rho_gt, agent_rho_rec, transient, agent_results_dir)
        print(f"Agent metrics: {metrics_agent}")
    except Exception as e:
        print(f"ERROR during agent evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\nEvaluating standard reconstruction...")
    try:
        metrics_std = evaluate_results(rho_gt, std_rho_rec, transient, std_results_dir)
        print(f"Standard metrics: {metrics_std}")
    except Exception as e:
        print(f"ERROR during standard evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics for comparison
    # Using PSNR as the primary metric (higher is better)
    score_agent = metrics_agent['PSNR']
    score_std = metrics_std['PSNR']
    
    print(f"\nScores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    
    # Also print other metrics
    print(f"SSIM  -> Agent: {metrics_agent['SSIM']:.4f}, Standard: {metrics_std['SSIM']:.4f}")
    print(f"CC    -> Agent: {metrics_agent['CC']:.4f}, Standard: {metrics_std['CC']:.4f}")
    print(f"RMSE  -> Agent: {metrics_agent['RMSE']:.6f}, Standard: {metrics_std['RMSE']:.6f}")
    
    # Determine success
    # PSNR: Higher is better
    # Allow 10% margin of error
    threshold = score_std * 0.9
    
    print(f"\nThreshold (90% of standard): {threshold:.4f}")
    
    if score_agent >= threshold:
        print("SUCCESS: Agent performance is acceptable!")
        sys.exit(0)
    else:
        print(f"FAILURE: Agent PSNR ({score_agent:.4f}) is below threshold ({threshold:.4f})")
        sys.exit(1)


if __name__ == "__main__":
    main()