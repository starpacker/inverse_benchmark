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
from skimage.metrics import structural_similarity as ssim_func


# Inject the referee function (evaluate_results) verbatim from Reference B
def evaluate_results(data, result, results_dir=None):
    """
    Evaluate reconstruction quality and optionally save results.
    
    Parameters
    ----------
    data : dict - output from load_and_preprocess_data
    result : dict - output from run_inversion
    results_dir : str or None - directory to save results (if None, skip saving)
    
    Returns
    -------
    metrics : dict - evaluation metrics
    """
    model = data['model']
    final_image = result['final_image']
    dirty_image = result['dirty_image']
    
    # Compute PSNR
    def compute_psnr(ref, test, data_range=None):
        if data_range is None:
            data_range = ref.max() - ref.min()
        mse = np.mean((ref.astype(float) - test.astype(float))**2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range**2 / mse)
    
    # Compute SSIM
    def compute_ssim(ref, test):
        data_range = ref.max() - ref.min()
        return ssim_func(ref, test, data_range=data_range)
    
    # Compute correlation coefficient
    def compute_cc(ref, test):
        return float(np.corrcoef(ref.ravel(), test.ravel())[0, 1])
    
    n_valid = int(np.sum(data['valid']))
    
    metrics = {
        "task": "suncasa_radio",
        "task_id": 199,
        "method": result['method_used'],
        "n_antennas": data['config']['n_ant'],
        "n_visibilities": n_valid,
        "image_size": data['config']['n'],
        "clean_iterations": len(result['clean_components']),
        "psnr": float(compute_psnr(model, final_image)),
        "ssim": float(compute_ssim(model, final_image)),
        "cc": float(compute_cc(model, final_image)),
        "rmse": float(np.sqrt(np.mean((model - final_image)**2))),
        "dirty_psnr": float(compute_psnr(model, dirty_image)),
        "clean_psnr": float(result['psnr_clean']),
        "wiener_psnr": float(result['psnr_wiener']),
    }
    
    print(f"  PSNR = {metrics['psnr']:.2f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")
    print(f"  CC   = {metrics['cc']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  Dirty PSNR = {metrics['dirty_psnr']:.2f} dB (baseline)")
    
    # Save results if directory provided
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")
        
        np.save(os.path.join(results_dir, "ground_truth.npy"), model)
        np.save(os.path.join(results_dir, "reconstruction.npy"), final_image)
        
        # Visualization
        u = data['u']
        v = data['v']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Task 199: Solar Radio Image Reconstruction (CLEAN)\n"
            f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | CC={metrics['cc']:.4f}",
            fontsize=14, fontweight='bold'
        )
        
        vmin, vmax = 0, model.max()
        
        # Row 1: Images
        ax = axes[0, 0]
        im = ax.imshow(model, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title('Ground Truth\n(Solar Radio Model)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[0, 1]
        im = ax.imshow(dirty_image, cmap='hot', origin='lower')
        ax.set_title('Dirty Image\n(with sidelobes)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[0, 2]
        im = ax.imshow(final_image, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title('CLEAN Reconstruction')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 2: UV coverage, error map, profiles
        ax = axes[1, 0]
        ax.scatter(u, v, s=0.1, c='blue', alpha=0.3)
        ax.set_xlabel('u (wavelengths)')
        ax.set_ylabel('v (wavelengths)')
        ax.set_title(f'(u,v) Coverage\n({len(u)} visibilities)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        error = np.abs(model - final_image)
        im = ax.imshow(error, cmap='viridis', origin='lower')
        ax.set_title('Error Map\n|GT - CLEAN|')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[1, 2]
        mid = model.shape[0] // 2
        ax.plot(model[mid, :], 'b-', lw=2, label='Ground Truth')
        ax.plot(dirty_image[mid, :], 'gray', alpha=0.5, label='Dirty')
        ax.plot(final_image[mid, :], 'r--', lw=2, label='CLEAN')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Brightness')
        ax.set_title('Central Profile Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = os.path.join(results_dir, "reconstruction_result.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIS] Saved → {vis_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/suncasa_radio_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Check if we have the main data file
    if not outer_data_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load the primary (outer) data
    outer_path = outer_data_files[0]
    print(f"\nLoading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs...")
    
    # Execute the agent's run_inversion
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR executing run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is chained execution
    if inner_data_files:
        # Pattern 2: Chained Execution
        print("\n=== Chained Execution Detected ===")
        inner_path = inner_data_files[0]
        print(f"Loading inner data from: {inner_path}")
        
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
        
        # Execute the operator returned by run_inversion
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR executing inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("\n=== Direct Execution ===")
        final_result = agent_output
        std_result = std_output
    
    # Get the input data for evaluation
    # The first argument to run_inversion is 'data' dict
    if args:
        input_data = args[0]
    else:
        input_data = kwargs.get('data', None)
    
    if input_data is None:
        print("ERROR: Could not find input data for evaluation!")
        sys.exit(1)
    
    # Evaluate results
    print("\n=== Evaluating Agent Result ===")
    try:
        agent_metrics = evaluate_results(input_data, final_result)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n=== Evaluating Standard Result ===")
    try:
        std_metrics = evaluate_results(input_data, std_result)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics for comparison
    # Using PSNR as the primary metric (higher is better)
    agent_psnr = agent_metrics['psnr']
    std_psnr = std_metrics['psnr']
    
    agent_ssim = agent_metrics['ssim']
    std_ssim = std_metrics['ssim']
    
    print(f"\n=== Comparison ===")
    print(f"Scores -> Agent PSNR: {agent_psnr:.4f}, Standard PSNR: {std_psnr:.4f}")
    print(f"Scores -> Agent SSIM: {agent_ssim:.4f}, Standard SSIM: {std_ssim:.4f}")
    
    # Determine success: PSNR and SSIM are "higher is better"
    # Allow 10% margin of error
    psnr_threshold = std_psnr * 0.9
    ssim_threshold = std_ssim * 0.9
    
    psnr_pass = agent_psnr >= psnr_threshold
    ssim_pass = agent_ssim >= ssim_threshold
    
    print(f"\nPSNR threshold (90%): {psnr_threshold:.4f} -> {'PASS' if psnr_pass else 'FAIL'}")
    print(f"SSIM threshold (90%): {ssim_threshold:.4f} -> {'PASS' if ssim_pass else 'FAIL'}")
    
    # Overall pass if both metrics are acceptable
    if psnr_pass and ssim_pass:
        print("\n=== TEST PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TEST FAILED ===")
        print(f"Agent performance degraded significantly from standard.")
        sys.exit(1)


if __name__ == "__main__":
    main()