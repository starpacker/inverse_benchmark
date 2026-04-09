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
import json
import matplotlib.pyplot as plt

# Inject the evaluate_results function (Reference B)
def evaluate_results(gt_norm, zf_norm, recon_norm, recon_metrics, method_name, 
                     zf_metrics, N, acceleration, results_dir):
    """
    Evaluate reconstruction results, create visualizations, and save outputs.
    
    Args:
        gt_norm: Normalized ground truth image
        zf_norm: Normalized zero-filled reconstruction
        recon_norm: Best reconstruction result
        recon_metrics: (PSNR, SSIM, RMSE) for reconstruction
        method_name: Name of the reconstruction method
        zf_metrics: (PSNR, SSIM, RMSE) for zero-filled baseline
        N: Image size
        acceleration: Acceleration factor
        results_dir: Directory to save results
    
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute error map
    error_map = np.abs(gt_norm - recon_norm)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(gt_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(zf_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Zero-Filled IFFT\nPSNR={zf_metrics[0]:.1f}dB, SSIM={zf_metrics[1]:.3f}', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'{method_name}\nPSNR={recon_metrics[0]:.1f}dB, SSIM={recon_metrics[1]:.3f}', fontsize=12)
    axes[2].axis('off')

    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.15)
    axes[3].set_title('Error Map (|GT - Recon|)', fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('Task 145: Deep Learning MRI Reconstruction', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_norm)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_norm)
    
    # Create metrics dictionary
    metrics_dict = {
        'task': 'direct_mri',
        'method': method_name,
        'PSNR': float(round(recon_metrics[0], 4)),
        'SSIM': float(round(recon_metrics[1], 4)),
        'RMSE': float(round(recon_metrics[2], 4)),
        'zero_filled_PSNR': float(round(zf_metrics[0], 4)),
        'zero_filled_SSIM': float(round(zf_metrics[1], 4)),
        'image_size': N,
        'acceleration': acceleration,
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved metrics.json")
    
    # Print summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print(f"  PSNR  : {recon_metrics[0]:.2f} dB {'PASS' if recon_metrics[0] > 15 else 'FAIL'}")
    print(f"  SSIM  : {recon_metrics[1]:.4f} {'PASS' if recon_metrics[1] > 0.5 else 'FAIL'}")
    print(f"  RMSE  : {recon_metrics[2]:.4f}")
    print(f"  Method: {method_name}")
    status = "PASS" if recon_metrics[0] > 15 and recon_metrics[1] > 0.5 else "FAIL"
    print(f"  Status: {status}")
    print("=" * 65)
    
    return metrics_dict


def main():
    # Data paths provided
    data_paths = ['/data/yjh/direct_mri_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    try:
        # Load the outer (main) data
        if outer_data_path is None:
            print("ERROR: No outer data file found!")
            sys.exit(1)
            
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent's implementation
        print("\n" + "=" * 65)
        print("Running agent's run_inversion...")
        print("=" * 65)
        
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_data_paths:
            # Chained execution pattern
            print("\nDetected chained execution pattern...")
            for inner_path in inner_data_paths:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                # Execute the operator returned by agent
                if callable(agent_output):
                    final_result = agent_output(*inner_args, **inner_kwargs)
                else:
                    final_result = agent_output
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Extract metrics from outputs
        # run_inversion returns: (recon_norm, recon_metrics, method_name)
        # where recon_metrics = (PSNR, SSIM, RMSE)
        
        print("\n" + "=" * 65)
        print("Comparing Results")
        print("=" * 65)
        
        if isinstance(final_result, tuple) and len(final_result) >= 3:
            agent_recon, agent_metrics, agent_method = final_result
            agent_psnr = agent_metrics[0]
            agent_ssim = agent_metrics[1]
            agent_rmse = agent_metrics[2]
            print(f"\nAgent results:")
            print(f"  Method: {agent_method}")
            print(f"  PSNR: {agent_psnr:.4f} dB")
            print(f"  SSIM: {agent_ssim:.4f}")
            print(f"  RMSE: {agent_rmse:.4f}")
        else:
            print(f"Unexpected agent output format: {type(final_result)}")
            agent_psnr = None
            agent_ssim = None
            agent_rmse = None
        
        if isinstance(std_result, tuple) and len(std_result) >= 3:
            std_recon, std_metrics, std_method = std_result
            std_psnr = std_metrics[0]
            std_ssim = std_metrics[1]
            std_rmse = std_metrics[2]
            print(f"\nStandard results:")
            print(f"  Method: {std_method}")
            print(f"  PSNR: {std_psnr:.4f} dB")
            print(f"  SSIM: {std_ssim:.4f}")
            print(f"  RMSE: {std_rmse:.4f}")
        else:
            print(f"Unexpected standard output format: {type(std_result)}")
            std_psnr = None
            std_ssim = None
            std_rmse = None
        
        # Verification logic
        # For PSNR and SSIM: higher is better
        # For RMSE: lower is better
        # We allow a margin of 10% degradation
        
        print("\n" + "=" * 65)
        print("Performance Verification")
        print("=" * 65)
        
        success = True
        margin = 0.90  # Allow 10% degradation
        
        if agent_psnr is not None and std_psnr is not None:
            psnr_threshold = std_psnr * margin
            psnr_pass = agent_psnr >= psnr_threshold
            print(f"\nPSNR Check: Agent={agent_psnr:.4f}, Standard={std_psnr:.4f}, Threshold={psnr_threshold:.4f}")
            print(f"  Result: {'PASS' if psnr_pass else 'FAIL'}")
            if not psnr_pass:
                success = False
        
        if agent_ssim is not None and std_ssim is not None:
            ssim_threshold = std_ssim * margin
            ssim_pass = agent_ssim >= ssim_threshold
            print(f"\nSSIM Check: Agent={agent_ssim:.4f}, Standard={std_ssim:.4f}, Threshold={ssim_threshold:.4f}")
            print(f"  Result: {'PASS' if ssim_pass else 'FAIL'}")
            if not ssim_pass:
                success = False
        
        if agent_rmse is not None and std_rmse is not None:
            # For RMSE, lower is better, so allow 10% increase
            rmse_threshold = std_rmse * (2 - margin)  # 1.10 * std_rmse
            rmse_pass = agent_rmse <= rmse_threshold
            print(f"\nRMSE Check: Agent={agent_rmse:.4f}, Standard={std_rmse:.4f}, Threshold={rmse_threshold:.4f}")
            print(f"  Result: {'PASS' if rmse_pass else 'FAIL'}")
            if not rmse_pass:
                success = False
        
        # Also check absolute thresholds from the task requirements
        # PSNR > 15 and SSIM > 0.5
        if agent_psnr is not None:
            abs_psnr_pass = agent_psnr > 15
            print(f"\nAbsolute PSNR Check (>15): {agent_psnr:.4f} -> {'PASS' if abs_psnr_pass else 'FAIL'}")
            if not abs_psnr_pass:
                success = False
        
        if agent_ssim is not None:
            abs_ssim_pass = agent_ssim > 0.5
            print(f"Absolute SSIM Check (>0.5): {agent_ssim:.4f} -> {'PASS' if abs_ssim_pass else 'FAIL'}")
            if not abs_ssim_pass:
                success = False
        
        print("\n" + "=" * 65)
        if success:
            print("OVERALL RESULT: PASS")
            print("=" * 65)
            sys.exit(0)
        else:
            print("OVERALL RESULT: FAIL")
            print("=" * 65)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()