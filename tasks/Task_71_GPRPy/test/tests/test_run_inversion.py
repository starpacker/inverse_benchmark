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

# Inject the referee function
def evaluate_results(ground_truth, reconstruction, x_traces, z_depth, t_axis, bscan_noisy, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        ground_truth: true reflectivity model (nx, nz)
        reconstruction: migrated/reconstructed image (nx, nz)
        x_traces: trace positions array
        z_depth: depth axis array
        t_axis: time axis array
        bscan_noisy: noisy B-scan data for visualization
        results_dir: directory to save outputs
    
    Returns:
        metrics: dictionary containing PSNR, SSIM, CC, RE, RMSE
    """
    # Normalize for comparison
    gt_n = ground_truth / max(ground_truth.max(), 1e-12)
    rec_n = reconstruction / max(reconstruction.max(), 1e-12)
    data_range = 1.0
    
    # Compute metrics
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse
    }
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), reconstruction)
    np.save(os.path.join(results_dir, "ground_truth.npy"), ground_truth)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Subsurface model
    axes[0, 0].imshow(ground_truth.T, aspect='auto', cmap='gray_r',
                       extent=[x_traces[0], x_traces[-1], z_depth[-1], z_depth[0]])
    axes[0, 0].set_title('True Reflectivity Model')
    axes[0, 0].set_xlabel('Position [m]')
    axes[0, 0].set_ylabel('Depth [m]')
    
    # B-scan
    clip = np.percentile(np.abs(bscan_noisy), 98)
    axes[0, 1].imshow(bscan_noisy.T, aspect='auto', cmap='RdBu_r', vmin=-clip, vmax=clip,
                       extent=[x_traces[0], x_traces[-1], t_axis[-1]*1e9, t_axis[0]*1e9])
    axes[0, 1].set_title('GPR B-Scan (noisy)')
    axes[0, 1].set_xlabel('Position [m]')
    axes[0, 1].set_ylabel('Two-way time [ns]')
    
    # Migrated image
    axes[1, 0].imshow(reconstruction.T, aspect='auto', cmap='gray_r',
                       extent=[x_traces[0], x_traces[-1], z_depth[-1], z_depth[0]])
    axes[1, 0].set_title('Kirchhoff Migration')
    axes[1, 0].set_xlabel('Position [m]')
    axes[1, 0].set_ylabel('Depth [m]')
    
    # Cross-section comparison
    mid = ground_truth.shape[0] // 2
    axes[1, 1].plot(z_depth, ground_truth[mid, :] / max(ground_truth[mid, :].max(), 1e-12),
                     'b-', lw=2, label='GT')
    axes[1, 1].plot(z_depth, reconstruction[mid, :] / max(reconstruction[mid, :].max(), 1e-12),
                     'r--', lw=2, label='Migrated')
    axes[1, 1].set_title(f'Trace {mid} Comparison')
    axes[1, 1].set_xlabel('Depth [m]')
    axes[1, 1].legend()
    
    fig.suptitle(
        f"GPRPy — GPR Migration\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/GPRPy_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    # Determine execution pattern
    is_chained = len(inner_files) > 0
    
    try:
        # Load outer data
        if not outer_files:
            print("ERROR: No outer data file found")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Chained execution pattern
            inner_path = inner_files[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            # Execute the returned operator
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print(f"Agent output shape: {final_result.shape if hasattr(final_result, 'shape') else type(final_result)}")
        print(f"Standard output shape: {std_result.shape if hasattr(std_result, 'shape') else type(std_result)}")
        
        # For evaluation, we need the ground truth and other parameters
        # The evaluate_results function expects: ground_truth, reconstruction, x_traces, z_depth, t_axis, bscan_noisy, results_dir
        # We need to extract these from the input data
        
        # From the function signature: run_inversion(bscan, x_traces, z_depth, dt, v_em)
        bscan = args[0] if len(args) > 0 else kwargs.get('bscan')
        x_traces = args[1] if len(args) > 1 else kwargs.get('x_traces')
        z_depth = args[2] if len(args) > 2 else kwargs.get('z_depth')
        dt = args[3] if len(args) > 3 else kwargs.get('dt')
        v_em = args[4] if len(args) > 4 else kwargs.get('v_em')
        
        # Create synthetic t_axis from bscan shape and dt
        nt = bscan.shape[1]
        t_axis = np.arange(nt) * dt
        
        # Use standard result as ground truth for comparison
        # This tests if the agent produces similar quality results
        ground_truth = std_result
        
        # Create results directories
        agent_results_dir = './agent_results'
        std_results_dir = './std_results'
        
        # Evaluate agent result (comparing agent reconstruction against standard as ground truth)
        print("Evaluating agent output...")
        metrics_agent = evaluate_results(
            ground_truth=ground_truth,
            reconstruction=final_result,
            x_traces=x_traces,
            z_depth=z_depth,
            t_axis=t_axis,
            bscan_noisy=bscan,
            results_dir=agent_results_dir
        )
        
        # Evaluate standard result (comparing standard against itself - should be perfect)
        print("Evaluating standard output...")
        metrics_std = evaluate_results(
            ground_truth=ground_truth,
            reconstruction=std_result,
            x_traces=x_traces,
            z_depth=z_depth,
            t_axis=t_axis,
            bscan_noisy=bscan,
            results_dir=std_results_dir
        )
        
        # Extract primary metrics for comparison
        psnr_agent = metrics_agent['PSNR']
        psnr_std = metrics_std['PSNR']
        ssim_agent = metrics_agent['SSIM']
        ssim_std = metrics_std['SSIM']
        cc_agent = metrics_agent['CC']
        cc_std = metrics_std['CC']
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Agent Metrics:")
        print(f"  PSNR: {psnr_agent:.2f} dB")
        print(f"  SSIM: {ssim_agent:.4f}")
        print(f"  CC:   {cc_agent:.4f}")
        print(f"  RE:   {metrics_agent['RE']:.4f}")
        print(f"  RMSE: {metrics_agent['RMSE']:.6f}")
        print(f"\nStandard Metrics:")
        print(f"  PSNR: {psnr_std:.2f} dB")
        print(f"  SSIM: {ssim_std:.4f}")
        print(f"  CC:   {cc_std:.4f}")
        print(f"  RE:   {metrics_std['RE']:.4f}")
        print(f"  RMSE: {metrics_std['RMSE']:.6f}")
        print(f"{'='*60}")
        
        # Since we're comparing agent output to standard output as ground truth,
        # high PSNR and SSIM indicate good agreement
        # Standard comparing to itself will have infinite PSNR (or very high) and SSIM=1
        
        # Check if results are nearly identical (direct comparison)
        if np.allclose(final_result, std_result, rtol=1e-5, atol=1e-8):
            print("\nRESULT: Agent output matches standard output exactly!")
            print("TEST PASSED")
            sys.exit(0)
        
        # If not exact, check if the reconstruction quality is acceptable
        # PSNR > 30 dB is typically considered good quality
        # SSIM > 0.9 is typically considered high similarity
        # CC > 0.9 indicates strong correlation
        
        psnr_threshold = 25.0  # dB
        ssim_threshold = 0.85
        cc_threshold = 0.85
        
        # For this case, since we compare agent to standard, we want high values
        if psnr_agent > psnr_threshold and ssim_agent > ssim_threshold and cc_agent > cc_threshold:
            print(f"\nRESULT: Agent performance is acceptable")
            print(f"  PSNR {psnr_agent:.2f} > {psnr_threshold} dB ✓")
            print(f"  SSIM {ssim_agent:.4f} > {ssim_threshold} ✓")
            print(f"  CC {cc_agent:.4f} > {cc_threshold} ✓")
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"\nRESULT: Agent performance is below threshold")
            if psnr_agent <= psnr_threshold:
                print(f"  PSNR {psnr_agent:.2f} <= {psnr_threshold} dB ✗")
            if ssim_agent <= ssim_threshold:
                print(f"  SSIM {ssim_agent:.4f} <= {ssim_threshold} ✗")
            if cc_agent <= cc_threshold:
                print(f"  CC {cc_agent:.4f} <= {cc_threshold} ✗")
            print("TEST FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Exception during testing")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()