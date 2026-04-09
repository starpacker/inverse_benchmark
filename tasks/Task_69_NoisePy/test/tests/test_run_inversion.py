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


def evaluate_results(data, inversion_results, results_dir):
    """
    Compute evaluation metrics, save results, and generate visualizations.
    
    Metrics:
        - PSNR: Peak Signal-to-Noise Ratio
        - SSIM: Structural Similarity Index
        - CC: Correlation Coefficient
        - RE: Relative Error
        - RMSE: Root Mean Square Error
    
    Args:
        data: dict containing dm_gt, stations, pairs, and grid info
        inversion_results: dict containing best_rec and inversion info
        results_dir: directory to save outputs
        
    Returns:
        dict with computed metrics
    """
    dm_gt = data['dm_gt']
    stations = data['stations']
    pairs = data['pairs']
    xmin = data['xmin']
    xmax = data['xmax']
    ymin = data['ymin']
    ymax = data['ymax']
    
    best_rec = inversion_results['best_rec']
    best_alpha = inversion_results['best_alpha']
    
    # Compute metrics
    data_range = dm_gt.max() - dm_gt.min()
    if data_range < 1e-12:
        data_range = 1.0
    
    mse = np.mean((dm_gt - best_rec)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(dm_gt, best_rec, data_range=data_range))
    cc = float(np.corrcoef(dm_gt.ravel(), best_rec.ravel())[0, 1])
    re = float(np.linalg.norm(dm_gt - best_rec) / max(np.linalg.norm(dm_gt), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse
    }
    
    # Save metrics and arrays
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), dm_gt)
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmax = max(np.abs(dm_gt).max(), np.abs(best_rec).max())
    extent = [xmin, xmax, ymin, ymax]
    
    # Ray coverage
    ax = axes[0, 0]
    for si, sj in pairs[:200]:  # Plot subset of rays
        ax.plot([stations[si, 0], stations[sj, 0]],
                [stations[si, 1], stations[sj, 1]],
                'b-', alpha=0.05, lw=0.5)
    ax.plot(stations[:, 0], stations[:, 1], 'r^', ms=5)
    ax.set_title(f'Ray Coverage ({len(pairs)} paths)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    
    # Ground truth
    im1 = axes[0, 1].imshow(dm_gt.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             origin='lower', extent=extent, aspect='equal')
    axes[0, 1].set_title('Ground Truth δc/c₀')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Reconstruction
    im2 = axes[1, 0].imshow(best_rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             origin='lower', extent=extent, aspect='equal')
    axes[1, 0].set_title('LSQR Reconstruction')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Error
    err = dm_gt - best_rec
    im3 = axes[1, 1].imshow(err.T, cmap='RdBu_r', origin='lower',
                             extent=extent, aspect='equal')
    axes[1, 1].set_title('Error')
    plt.colorbar(im3, ax=axes[1, 1])
    
    fig.suptitle(
        f"NoisePy — Ambient Noise Tomography\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f} | "
        f"RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/NoisePy_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found.")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs...")
        
        # Execute the target function
        agent_output = run_inversion(*args, **kwargs)
        
        # Determine if chained execution is needed
        if inner_data_files:
            # Chained execution pattern
            print("Chained execution detected...")
            inner_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute chained call
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("Direct execution pattern...")
            final_result = agent_output
            std_result = std_output
        
        print(f"Agent output type: {type(final_result)}")
        print(f"Standard output type: {type(std_result)}")
        
        # For evaluate_results, we need the original 'data' dict with dm_gt, stations, etc.
        # The 'data' is passed as the first argument to run_inversion
        # Extract it from args
        if len(args) >= 1:
            input_data = args[0]
        else:
            input_data = kwargs.get('data', {})
        
        # We need to create a proper data dict for evaluate_results
        # The input_data should contain: dm_gt, stations, pairs, grid info
        # Let's check what's available
        print(f"Input data keys: {input_data.keys() if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Build the evaluation data dict
        # The run_inversion input data contains: G, nx, ny, c0, dm_gt_flat
        # We need to construct dm_gt from dm_gt_flat using nx, ny
        eval_data = {}
        
        if isinstance(input_data, dict):
            nx = input_data.get('nx', None)
            ny = input_data.get('ny', None)
            dm_gt_flat = input_data.get('dm_gt_flat', None)
            
            if dm_gt_flat is not None and nx is not None and ny is not None:
                # Reshape dm_gt_flat to 2D
                eval_data['dm_gt'] = dm_gt_flat.reshape(nx, ny)
            elif 'dm_gt' in input_data:
                eval_data['dm_gt'] = input_data['dm_gt']
            
            # Get other required fields, use defaults if not available
            eval_data['stations'] = input_data.get('stations', np.random.rand(10, 2) * 100)
            eval_data['pairs'] = input_data.get('pairs', [(0, 1)] * 10)
            eval_data['xmin'] = input_data.get('xmin', 0)
            eval_data['xmax'] = input_data.get('xmax', 100)
            eval_data['ymin'] = input_data.get('ymin', 0)
            eval_data['ymax'] = input_data.get('ymax', 100)
        
        # Create output directories
        agent_results_dir = '/tmp/agent_results'
        std_results_dir = '/tmp/std_results'
        
        # Evaluate agent results
        print("\nEvaluating agent results...")
        agent_metrics = evaluate_results(eval_data, final_result, agent_results_dir)
        print(f"Agent metrics: {agent_metrics}")
        
        # Evaluate standard results
        print("\nEvaluating standard results...")
        std_metrics = evaluate_results(eval_data, std_result, std_results_dir)
        print(f"Standard metrics: {std_metrics}")
        
        # Compare metrics
        # Primary metric: CC (Correlation Coefficient) - higher is better
        # Also check PSNR (higher is better) and RE (lower is better)
        
        agent_cc = agent_metrics['CC']
        std_cc = std_metrics['CC']
        
        agent_psnr = agent_metrics['PSNR']
        std_psnr = std_metrics['PSNR']
        
        agent_re = agent_metrics['RE']
        std_re = std_metrics['RE']
        
        print(f"\n{'='*60}")
        print(f"Scores Comparison:")
        print(f"  CC    -> Agent: {agent_cc:.6f}, Standard: {std_cc:.6f}")
        print(f"  PSNR  -> Agent: {agent_psnr:.2f} dB, Standard: {std_psnr:.2f} dB")
        print(f"  RE    -> Agent: {agent_re:.6f}, Standard: {std_re:.6f}")
        print(f"{'='*60}")
        
        # Verification with tolerance
        # For CC and PSNR: higher is better, allow 10% margin
        # For RE: lower is better, allow 10% margin
        
        tolerance = 0.10  # 10% tolerance
        
        # Check CC (higher is better)
        cc_threshold = std_cc * (1 - tolerance)
        cc_pass = agent_cc >= cc_threshold
        
        # Check PSNR (higher is better)
        psnr_threshold = std_psnr * (1 - tolerance)
        psnr_pass = agent_psnr >= psnr_threshold
        
        # Check RE (lower is better)
        re_threshold = std_re * (1 + tolerance)
        re_pass = agent_re <= re_threshold
        
        print(f"\nVerification Results:")
        print(f"  CC pass (>= {cc_threshold:.6f}): {cc_pass}")
        print(f"  PSNR pass (>= {psnr_threshold:.2f}): {psnr_pass}")
        print(f"  RE pass (<= {re_threshold:.6f}): {re_pass}")
        
        # Overall pass if at least CC and one other metric pass
        overall_pass = cc_pass and (psnr_pass or re_pass)
        
        if overall_pass:
            print("\n✓ Performance verification PASSED")
            sys.exit(0)
        else:
            print("\n✗ Performance verification FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()