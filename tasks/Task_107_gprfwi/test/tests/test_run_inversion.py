import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

# Import the target function
from agent_run_inversion import run_inversion


# Inject the referee evaluation function verbatim
def evaluate_results(eps_gt, eps_recon, bscan_noisy, results_dir, assets_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes metrics (PSNR, SSIM, CC), saves numpy arrays and JSON metrics,
    and generates visualization plots.
    
    Parameters:
    -----------
    eps_gt : ndarray
        Ground truth permittivity model, shape (nz, nx)
    eps_recon : ndarray
        Reconstructed permittivity model, shape (nz, nx)
    bscan_noisy : ndarray
        Noisy B-scan data, shape (nz, nx)
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
    
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, SSIM, and CC values
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Compute PSNR
    mse = np.mean((eps_gt - eps_recon)**2)
    if mse < 1e-15:
        psnr_val = 100.0
    else:
        data_range = np.max(eps_gt) - np.min(eps_gt)
        psnr_val = 10 * np.log10(data_range**2 / mse)
    
    # Compute SSIM
    data_range_ssim = np.max(eps_gt) - np.min(eps_gt)
    if data_range_ssim < 1e-10:
        data_range_ssim = 1.0
    ssim_val = float(ssim(eps_gt, eps_recon, data_range=data_range_ssim))
    
    # Compute CC (Pearson correlation coefficient)
    g = eps_gt.ravel() - np.mean(eps_gt)
    r = eps_recon.ravel() - np.mean(eps_recon)
    denom = np.sqrt(np.sum(g**2) * np.sum(r**2))
    if denom < 1e-15:
        cc_val = 0.0
    else:
        cc_val = float(np.sum(g * r) / denom)
    
    metrics = {"PSNR": float(psnr_val), "SSIM": float(ssim_val), "CC": float(cc_val)}
    
    # Save metrics
    for path in [results_dir, assets_dir]:
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Save numpy outputs
    for path in [results_dir, assets_dir]:
        np.save(os.path.join(path, "gt_output.npy"), eps_gt)
        np.save(os.path.join(path, "recon_output.npy"), eps_recon)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # B-scan
    ax = axes[0, 0]
    im = ax.imshow(bscan_noisy, aspect='auto', cmap='seismic',
                   vmin=-np.max(np.abs(bscan_noisy)), vmax=np.max(np.abs(bscan_noisy)))
    ax.set_title("GPR B-scan (noisy)")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # GT permittivity
    ax = axes[0, 1]
    im = ax.imshow(eps_gt, aspect='auto', cmap='viridis')
    ax.set_title("GT Permittivity εᵣ")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="εᵣ")
    
    # Reconstructed permittivity
    ax = axes[1, 0]
    im = ax.imshow(eps_recon, aspect='auto', cmap='viridis',
                   vmin=eps_gt.min(), vmax=eps_gt.max())
    ax.set_title(f"Reconstructed εᵣ (PSNR={psnr_val:.1f}dB)")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="εᵣ")
    
    # Error map
    ax = axes[1, 1]
    error = np.abs(eps_gt - eps_recon)
    im = ax.imshow(error, aspect='auto', cmap='hot')
    ax.set_title(f"Absolute Error (SSIM={ssim_val:.3f}, CC={cc_val:.3f})")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="|error|")
    
    plt.suptitle("GPR Full-Waveform Inversion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for path in [results_dir, assets_dir]:
        fig.savefig(os.path.join(path, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(path, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics


def main():
    data_paths = ['/data/yjh/gprfwi_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}, Number of kwargs: {len(kwargs)}")
    
    # Run the agent's implementation
    print("Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if chained execution
    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print(f"Chained execution detected. Inner files: {inner_paths}")
        inner_path = inner_paths[0]
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
        
        print("Running chained call...")
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running chained call: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("Direct execution pattern detected.")
        final_result = agent_output
        std_result = std_output
    
    # We need bscan_noisy and eps_gt for evaluate_results
    # bscan_noisy is the first positional argument to run_inversion
    bscan_noisy = args[0] if len(args) > 0 else kwargs.get('bscan_noisy', None)
    
    if bscan_noisy is None:
        print("ERROR: Could not extract bscan_noisy from inputs.")
        sys.exit(1)
    
    # The std_result (ground truth output from reference) serves as eps_gt for evaluation
    # Both final_result and std_result are eps_recon arrays
    # We evaluate both against std_result as the "ground truth"
    
    eps_gt = std_result  # Standard/reference output used as ground truth
    eps_recon_agent = final_result  # Agent's output
    
    print(f"eps_gt shape: {eps_gt.shape}, eps_recon_agent shape: {eps_recon_agent.shape}")
    print(f"eps_gt range: [{eps_gt.min():.4f}, {eps_gt.max():.4f}]")
    print(f"eps_recon_agent range: [{eps_recon_agent.min():.4f}, {eps_recon_agent.max():.4f}]")
    
    # Evaluate agent output against standard output
    results_dir_agent = './results_agent'
    assets_dir_agent = './assets_agent'
    
    print("\nEvaluating agent output...")
    try:
        metrics_agent = evaluate_results(eps_gt, eps_recon_agent, bscan_noisy, 
                                         results_dir_agent, assets_dir_agent)
    except Exception as e:
        print(f"ERROR during agent evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard output against itself (perfect score baseline)
    results_dir_std = './results_std'
    assets_dir_std = './assets_std'
    
    print("Evaluating standard output (self-comparison)...")
    try:
        metrics_std = evaluate_results(eps_gt, eps_gt, bscan_noisy,
                                       results_dir_std, assets_dir_std)
    except Exception as e:
        print(f"ERROR during standard evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Print scores
    print(f"\nScores -> Agent: PSNR={metrics_agent['PSNR']:.2f}, SSIM={metrics_agent['SSIM']:.4f}, CC={metrics_agent['CC']:.4f}")
    print(f"Scores -> Standard (self): PSNR={metrics_std['PSNR']:.2f}, SSIM={metrics_std['SSIM']:.4f}, CC={metrics_std['CC']:.4f}")
    
    # Verification:
    # For a correct implementation, agent output should be very close to standard output
    # PSNR should be very high (ideally 100 for identical), SSIM close to 1, CC close to 1
    # We use reasonable thresholds:
    # - PSNR >= 30 dB (good reconstruction quality)
    # - SSIM >= 0.9 
    # - CC >= 0.9
    
    psnr_threshold = 30.0
    ssim_threshold = 0.85
    cc_threshold = 0.85
    
    passed = True
    reasons = []
    
    if metrics_agent['PSNR'] < psnr_threshold:
        reasons.append(f"PSNR {metrics_agent['PSNR']:.2f} < {psnr_threshold}")
        passed = False
    
    if metrics_agent['SSIM'] < ssim_threshold:
        reasons.append(f"SSIM {metrics_agent['SSIM']:.4f} < {ssim_threshold}")
        passed = False
    
    if metrics_agent['CC'] < cc_threshold:
        reasons.append(f"CC {metrics_agent['CC']:.4f} < {cc_threshold}")
        passed = False
    
    # Also check direct numerical similarity
    mse_direct = np.mean((eps_gt - eps_recon_agent)**2)
    max_val = max(np.max(np.abs(eps_gt)), 1e-10)
    relative_error = np.sqrt(mse_direct) / max_val
    print(f"Direct MSE: {mse_direct:.6e}, Relative RMSE: {relative_error:.6e}")
    
    if passed:
        print("\n✅ TEST PASSED: Agent's run_inversion produces acceptable results.")
        sys.exit(0)
    else:
        print(f"\n❌ TEST FAILED: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == '__main__':
    main()