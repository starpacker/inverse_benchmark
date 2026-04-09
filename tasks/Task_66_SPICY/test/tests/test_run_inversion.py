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
from skimage.metrics import structural_similarity as ssim_fn

# Inject the referee (evaluation function)
def evaluate_results(data_dict, result_dict, results_dir):
    """
    Evaluate reconstruction quality and generate visualization.
    
    Parameters:
        data_dict: Dictionary containing ground truth and grid data
        result_dict: Dictionary containing reconstructed pressure
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary of quality metrics
    """
    p_gt = data_dict['p_gt']
    p_rec = result_dict['p_rec']
    xx = data_dict['xx']
    yy = data_dict['yy']
    u_noisy = data_dict['u_noisy']
    v_noisy = data_dict['v_noisy']
    
    # Compute metrics (mean-removed)
    p_gt_zm = p_gt - p_gt.mean()
    p_rec_zm = p_rec - p_rec.mean()
    data_range = p_gt_zm.max() - p_gt_zm.min()
    if data_range < 1e-12:
        data_range = 1.0
    
    mse = np.mean((p_gt_zm - p_rec_zm)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(p_gt_zm, p_rec_zm, data_range=data_range))
    cc = float(np.corrcoef(p_gt_zm.ravel(), p_rec_zm.ravel())[0, 1])
    re = float(np.linalg.norm(p_gt_zm - p_rec_zm) / max(np.linalg.norm(p_gt_zm), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse,
        "method_used": result_dict['method_used']
    }
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), p_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), p_gt)
    
    # Visualization
    vmax = max(np.abs(p_gt_zm).max(), np.abs(p_rec_zm).max())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Velocity magnitude
    speed = np.sqrt(u_noisy**2 + v_noisy**2)
    im0 = axes[0, 0].contourf(xx, yy, speed, levels=30, cmap='viridis')
    axes[0, 0].set_title('Velocity Magnitude |V|')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # GT pressure
    im1 = axes[0, 1].contourf(xx, yy, p_gt_zm, levels=30, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Ground Truth Pressure')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Reconstructed pressure
    im2 = axes[1, 0].contourf(xx, yy, p_rec_zm, levels=30, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Reconstructed Pressure')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Error
    err = p_gt_zm - p_rec_zm
    im3 = axes[1, 1].contourf(xx, yy, err, levels=30, cmap='RdBu_r')
    axes[1, 1].set_title('Error (GT - Recon)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    fig.suptitle(
        f"SPICY — Pressure from PIV Reconstruction\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/SPICY_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze data paths to determine execution pattern
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    try:
        # Load outer data
        print(f"Loading outer data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'unknown')
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Function: {func_name}")
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Execute the agent function
        print("Running agent function: run_inversion")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check for chained execution pattern
        if inner_data_paths:
            # Chained execution: agent_output is an operator/function
            print(f"Detected chained execution with {len(inner_data_paths)} inner data file(s)")
            
            for inner_path in inner_data_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                # Execute the operator
                final_result = agent_output(*inner_args, **inner_kwargs)
                
                # For inner execution, we need to get data_dict from inner_args
                # This depends on the specific function signature
                data_dict = inner_args[0] if inner_args else inner_kwargs.get('data_dict', None)
        else:
            # Direct execution
            print("Direct execution pattern detected")
            final_result = agent_output
            std_result = std_output
            
            # data_dict is the first argument (input to run_inversion)
            data_dict = args[0] if args else kwargs.get('data_dict', None)
        
        # Create results directories
        results_dir_agent = './results_agent'
        results_dir_std = './results_std'
        
        # Evaluate agent results
        print("\nEvaluating agent results...")
        metrics_agent = evaluate_results(data_dict, final_result, results_dir_agent)
        print(f"Agent metrics: {metrics_agent}")
        
        # Evaluate standard results
        print("\nEvaluating standard results...")
        metrics_std = evaluate_results(data_dict, std_result, results_dir_std)
        print(f"Standard metrics: {metrics_std}")
        
        # Extract primary metrics for comparison
        # Higher is better for: PSNR, SSIM, CC
        # Lower is better for: RE, RMSE
        
        psnr_agent = metrics_agent['PSNR']
        psnr_std = metrics_std['PSNR']
        
        ssim_agent = metrics_agent['SSIM']
        ssim_std = metrics_std['SSIM']
        
        cc_agent = metrics_agent['CC']
        cc_std = metrics_std['CC']
        
        re_agent = metrics_agent['RE']
        re_std = metrics_std['RE']
        
        print(f"\n=== Performance Comparison ===")
        print(f"PSNR  -> Agent: {psnr_agent:.4f}, Standard: {psnr_std:.4f}")
        print(f"SSIM  -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
        print(f"CC    -> Agent: {cc_agent:.4f}, Standard: {cc_std:.4f}")
        print(f"RE    -> Agent: {re_agent:.4f}, Standard: {re_std:.4f}")
        
        # Determine success with a margin of error
        # For PSNR, SSIM, CC: agent should be at least 90% of standard
        # For RE: agent should be at most 110% of standard
        margin = 0.90  # 10% margin
        
        success = True
        
        # Check PSNR (higher is better)
        if psnr_agent < psnr_std * margin:
            print(f"WARNING: PSNR degraded significantly ({psnr_agent:.4f} < {psnr_std * margin:.4f})")
            success = False
        
        # Check SSIM (higher is better)
        if ssim_agent < ssim_std * margin:
            print(f"WARNING: SSIM degraded significantly ({ssim_agent:.4f} < {ssim_std * margin:.4f})")
            success = False
        
        # Check CC (higher is better, can be negative so handle carefully)
        if cc_std > 0:
            if cc_agent < cc_std * margin:
                print(f"WARNING: CC degraded significantly ({cc_agent:.4f} < {cc_std * margin:.4f})")
                success = False
        else:
            # If standard CC is negative or zero, just check agent is not much worse
            if cc_agent < cc_std - 0.1:
                print(f"WARNING: CC degraded significantly ({cc_agent:.4f} < {cc_std - 0.1:.4f})")
                success = False
        
        # Check RE (lower is better)
        if re_agent > re_std * (1.0 / margin):
            print(f"WARNING: RE degraded significantly ({re_agent:.4f} > {re_std * (1.0 / margin):.4f})")
            success = False
        
        if success:
            print("\n=== TEST PASSED ===")
            print("Agent performance is within acceptable range of standard.")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            print("Agent performance degraded significantly compared to standard.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()