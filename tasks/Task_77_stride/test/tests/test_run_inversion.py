import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim_fn

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the referee function (evaluate_results)
def evaluate_results(c_gt, inversion_results, transducers, domain_size, results_dir):
    """
    Evaluate reconstruction quality and select the best result.
    Compute metrics, generate visualizations, and save results.
    
    Args:
        c_gt: ground truth sound speed (nx, ny)
        inversion_results: dict from run_inversion
        transducers: transducer positions
        domain_size: physical domain size
        results_dir: directory to save results
        
    Returns:
        dict containing:
            - metrics: quality metrics (PSNR, SSIM, CC, RE, RMSE)
            - best_alpha: best regularisation parameter
            - c_rec: best reconstruction
    """
    all_results = inversion_results['all_results']
    alpha_list = inversion_results['alpha_list']

    best_cc = -1
    best_rec = None
    best_alpha = alpha_list[0]

    print("\n[INVERSION RESULTS]")
    for alpha in alpha_list:
        c_rec = all_results[alpha]
        cc_val = float(np.corrcoef(c_gt.ravel(), c_rec.ravel())[0, 1])
        print(f"  α={alpha:7.2f} → CC={cc_val:.4f}")
        if cc_val > best_cc:
            best_cc = cc_val
            best_alpha = alpha
            best_rec = c_rec

    print(f"  → Best α={best_alpha} with CC={best_cc:.4f}")

    # Compute metrics
    data_range = c_gt.max() - c_gt.min()
    if data_range < 1e-12:
        data_range = 1.0
    mse = np.mean((c_gt - best_rec)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(c_gt, best_rec, data_range=data_range))
    cc = float(np.corrcoef(c_gt.ravel(), best_rec.ravel())[0, 1])
    re = float(np.linalg.norm(c_gt - best_rec) / max(np.linalg.norm(c_gt), 1e-12))
    rmse = float(np.sqrt(mse))

    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}

    print("\n[EVALUATION METRICS]")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), c_gt)

    # Visualization
    nx, ny = c_gt.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    extent = [-domain_size / 2 * 1000, domain_size / 2 * 1000,
              -domain_size / 2 * 1000, domain_size / 2 * 1000]

    vmin = min(c_gt.min(), best_rec.min())
    vmax = max(c_gt.max(), best_rec.max())

    n_transducers = len(transducers)

    # GT
    im0 = axes[0, 0].imshow(c_gt.T, cmap='jet', vmin=vmin, vmax=vmax,
                              origin='lower', extent=extent)
    axes[0, 0].plot(transducers[:, 0] * 1000, transducers[:, 1] * 1000,
                     'k.', ms=3)
    axes[0, 0].set_title('Ground Truth c(x,y) [m/s]')
    axes[0, 0].set_xlabel('x [mm]')
    axes[0, 0].set_ylabel('y [mm]')
    plt.colorbar(im0, ax=axes[0, 0])

    # Reconstruction
    im1 = axes[0, 1].imshow(best_rec.T, cmap='jet', vmin=vmin, vmax=vmax,
                              origin='lower', extent=extent)
    axes[0, 1].plot(transducers[:, 0] * 1000, transducers[:, 1] * 1000,
                     'k.', ms=3)
    axes[0, 1].set_title('LSQR Reconstruction')
    plt.colorbar(im1, ax=axes[0, 1])

    # Error
    err = c_gt - best_rec
    im2 = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent)
    axes[1, 0].set_title('Error (GT - Recon)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Profile
    mid = c_gt.shape[0] // 2
    x_mm = np.linspace(-domain_size / 2, domain_size / 2, c_gt.shape[0]) * 1000
    axes[1, 1].plot(x_mm, c_gt[mid, :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(x_mm, best_rec[mid, :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title('Central Profile')
    axes[1, 1].set_xlabel('x [mm]')
    axes[1, 1].set_ylabel('Speed [m/s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle(
        f"stride — USCT Sound-Speed Tomography ({n_transducers} transducers)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'metrics': metrics,
        'best_alpha': best_alpha,
        'c_rec': best_rec
    }


def main():
    # Data paths provided
    data_paths = ['/data/yjh/stride_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Categorize files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Found {len(outer_files)} outer file(s) and {len(inner_files)} inner file(s)")
    
    # Load outer (primary) data
    if not outer_files:
        print("[ERROR] No primary data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"[INFO] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract arguments
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Function: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Run agent function
    print("\n[INFO] Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("[INFO] Agent function completed successfully")
    
    # Check if this is chained execution (closure/factory pattern)
    if inner_files:
        # Chained execution pattern
        inner_path = inner_files[0]
        print(f"\n[INFO] Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        print("[INFO] Executing returned operator with inner data...")
        try:
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"[ERROR] Operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        final_agent_result = agent_output
        std_result = std_output
    
    # For evaluation, we need additional context (c_gt, transducers, domain_size)
    # These should be available in the stored data or we need to extract them
    # Based on the run_inversion signature, we have: G, dt_noisy, nx, ny, c0, alpha_list
    
    # The inversion results are dicts with 'all_results', 'alpha_list', 'nx', 'ny'
    # To evaluate, we need c_gt (ground truth), transducers, and domain_size
    # These are typically provided externally or stored alongside
    
    # Since we don't have c_gt directly, we'll compare the reconstruction quality
    # by checking if both produce similar results
    
    print("\n[INFO] Comparing agent output with standard output...")
    
    # Verify output structure
    if not isinstance(final_agent_result, dict):
        print(f"[ERROR] Agent output is not a dict: {type(final_agent_result)}")
        sys.exit(1)
    
    if not isinstance(std_result, dict):
        print(f"[ERROR] Standard output is not a dict: {type(std_result)}")
        sys.exit(1)
    
    # Check required keys
    required_keys = ['all_results', 'alpha_list', 'nx', 'ny']
    for key in required_keys:
        if key not in final_agent_result:
            print(f"[ERROR] Missing key '{key}' in agent output")
            sys.exit(1)
        if key not in std_result:
            print(f"[ERROR] Missing key '{key}' in standard output")
            sys.exit(1)
    
    # Compare alpha_list
    agent_alphas = final_agent_result['alpha_list']
    std_alphas = std_result['alpha_list']
    print(f"[INFO] Agent alpha_list: {agent_alphas}")
    print(f"[INFO] Standard alpha_list: {std_alphas}")
    
    # Compare reconstructions for each alpha
    agent_all = final_agent_result['all_results']
    std_all = std_result['all_results']
    
    print("\n[INFO] Comparing reconstructions per alpha...")
    
    total_cc = 0
    count = 0
    max_diff = 0
    
    for alpha in std_alphas:
        if alpha not in agent_all:
            print(f"[WARN] Alpha {alpha} missing from agent results")
            continue
        if alpha not in std_all:
            print(f"[WARN] Alpha {alpha} missing from standard results")
            continue
        
        agent_rec = agent_all[alpha]
        std_rec = std_all[alpha]
        
        # Compute correlation coefficient between reconstructions
        cc = np.corrcoef(agent_rec.ravel(), std_rec.ravel())[0, 1]
        diff = np.abs(agent_rec - std_rec).max()
        mean_diff = np.abs(agent_rec - std_rec).mean()
        
        total_cc += cc
        count += 1
        max_diff = max(max_diff, diff)
        
        print(f"  α={alpha:7.2f}: CC={cc:.6f}, MaxDiff={diff:.6f}, MeanDiff={mean_diff:.6f}")
    
    if count == 0:
        print("[ERROR] No common alphas found for comparison")
        sys.exit(1)
    
    avg_cc = total_cc / count
    print(f"\n[INFO] Average CC across all alphas: {avg_cc:.6f}")
    print(f"[INFO] Maximum difference: {max_diff:.6f}")
    
    # Determine success based on correlation
    # High CC (close to 1.0) means good agreement
    CC_THRESHOLD = 0.95  # Require 95% correlation
    
    print(f"\n[VERIFICATION]")
    print(f"  Average Correlation: {avg_cc:.6f}")
    print(f"  Threshold: {CC_THRESHOLD}")
    
    if avg_cc >= CC_THRESHOLD:
        print(f"\n[SUCCESS] Agent output matches standard output (CC >= {CC_THRESHOLD})")
        sys.exit(0)
    else:
        print(f"\n[FAILURE] Agent output differs significantly from standard (CC < {CC_THRESHOLD})")
        sys.exit(1)


if __name__ == "__main__":
    main()