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
from scipy.ndimage import uniform_filter, gaussian_filter
import warnings
warnings.filterwarnings('ignore')


# Inject the referee evaluation function
def evaluate_results(kappa_true, kE, kB, g1_true, g2_true, g1_obs, g2_obs, params, results_dir):
    """
    Evaluate mass mapping quality and save results.
    
    Args:
        kappa_true: True convergence map
        kE: Reconstructed E-mode convergence
        kB: Reconstructed B-mode convergence
        g1_true: True shear component 1
        g2_true: True shear component 2
        g1_obs: Observed shear component 1
        g2_obs: Observed shear component 2
        params: Dictionary of parameters
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Remove mean (mass sheet degeneracy)
    gt = kappa_true - kappa_true.mean()
    recon = kE - kE.mean()
    
    # PSNR
    mse = np.mean((gt - recon)**2)
    data_range = gt.max() - gt.min()
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # SSIM
    def ssim_2d(img1, img2, win_size=7):
        C1 = (0.01 * data_range)**2
        C2 = (0.03 * data_range)**2
        mu1 = uniform_filter(img1, win_size)
        mu2 = uniform_filter(img2, win_size)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu12 = mu1 * mu2
        sigma1_sq = uniform_filter(img1**2, win_size) - mu1_sq
        sigma2_sq = uniform_filter(img2**2, win_size) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, win_size) - mu12
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    ssim = ssim_2d(gt, recon)
    
    # Correlation coefficient
    cc = np.corrcoef(gt.flatten(), recon.flatten())[0, 1]
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # Relative error
    re = rmse / np.sqrt(np.mean(gt**2)) if np.mean(gt**2) > 0 else float('inf')
    
    # Peak recovery
    gt_smooth = gaussian_filter(gt, sigma=3)
    recon_smooth = gaussian_filter(recon, sigma=3)
    gt_peak = np.unravel_index(np.argmax(gt_smooth), gt_smooth.shape)
    recon_peak = np.unravel_index(np.argmax(recon_smooth), recon_smooth.shape)
    peak_offset = np.sqrt((gt_peak[0] - recon_peak[0])**2 + (gt_peak[1] - recon_peak[1])**2)
    
    metrics = {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'cc': float(cc),
        'rmse': float(rmse),
        'relative_error': float(re),
        'peak_offset_pixels': float(peak_offset),
    }
    
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] SSIM = {metrics['ssim']:.6f}")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMSE = {metrics['rmse']:.6f}")
    print(f"[EVAL] Relative Error = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Peak offset = {metrics['peak_offset_pixels']:.2f} pixels")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    ny, nx = kappa_true.shape
    np.save(os.path.join(results_dir, "input.npy"), np.stack([g1_obs, g2_obs]))
    np.save(os.path.join(results_dir, "ground_truth.npy"), kappa_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), kE)
    print(f"[SAVE] Input shape: (2, {ny}, {nx}) → input.npy")
    print(f"[SAVE] GT shape: {kappa_true.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {kE.shape} → reconstruction.npy")
    
    # Visualization
    nx = params['nx']
    ny = params['ny']
    pixel_size = params['pixel_size']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    extent = [-nx / 2 * pixel_size, nx / 2 * pixel_size, -ny / 2 * pixel_size, ny / 2 * pixel_size]
    
    # (a) True convergence
    ax = axes[0, 0]
    im = ax.imshow(kappa_true, cmap='hot', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('True Convergence κ')
    plt.colorbar(im, ax=ax, label='κ')
    
    # (b) Observed shear field
    ax = axes[0, 1]
    gamma_mag = np.sqrt(g1_obs**2 + g2_obs**2)
    im2 = ax.imshow(gamma_mag, cmap='viridis', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('Observed |γ| (noisy)')
    plt.colorbar(im2, ax=ax, label='|γ|')
    
    # (c) KS93 reconstructed convergence (E-mode)
    ax = axes[0, 2]
    im3 = ax.imshow(kE, cmap='hot', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('KS93 Reconstructed κ_E')
    plt.colorbar(im3, ax=ax, label='κ_E')
    
    # (d) Error map
    ax = axes[1, 0]
    error = gt - recon
    vmax_err = np.max(np.abs(error)) * 0.8
    im4 = ax.imshow(error, cmap='seismic', origin='lower', extent=extent,
                    vmin=-vmax_err, vmax=vmax_err)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('Error (GT - Recon)')
    plt.colorbar(im4, ax=ax, label='Δκ')
    
    # (e) B-mode
    ax = axes[1, 1]
    im5 = ax.imshow(kB, cmap='RdBu_r', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('B-mode κ_B (noise diagnostic)')
    plt.colorbar(im5, ax=ax, label='κ_B')
    
    # (f) Scatter: GT vs Recon
    ax = axes[1, 2]
    ax.scatter(gt.flatten(), recon.flatten(), s=1, alpha=0.3, c='steelblue')
    lim = max(np.abs(gt).max(), np.abs(recon).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Identity')
    ax.set_xlabel('True κ (zero-mean)')
    ax.set_ylabel('Recon κ (zero-mean)')
    ax.set_title(f'True vs Recon (CC={metrics["cc"]:.4f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"lenspack — Kaiser-Squires Weak Lensing Mass Mapping\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"CC={metrics['cc']:.4f} | RMSE={metrics['rmse']:.6f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/lenspack_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    # We need to load the standard data and also need extra context (kappa_true, params, etc.)
    # for evaluation. This data might need to be generated or loaded from additional files.
    
    # Load the outer data
    if not outer_files:
        print("[ERROR] No outer data file found!")
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
    
    # Extract function inputs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(args)}")
    print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
    
    # Run agent function
    print("\n[RUN] Executing agent run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have chained execution
    final_agent_result = agent_output
    final_std_result = std_output
    
    if inner_files:
        # Chained execution pattern
        print("\n[INFO] Detected chained execution pattern")
        inner_path = inner_files[0]
        print(f"[INFO] Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        
        # Execute the returned operator
        print("[RUN] Executing agent output as operator...")
        try:
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"[ERROR] Agent operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        final_std_result = inner_data.get('output', None)
    
    # Unpack results (kE, kB)
    print("\n[INFO] Unpacking results...")
    
    if isinstance(final_agent_result, tuple) and len(final_agent_result) == 2:
        agent_kE, agent_kB = final_agent_result
    else:
        print(f"[ERROR] Unexpected agent result format: {type(final_agent_result)}")
        sys.exit(1)
    
    if isinstance(final_std_result, tuple) and len(final_std_result) == 2:
        std_kE, std_kB = final_std_result
    else:
        print(f"[ERROR] Unexpected standard result format: {type(final_std_result)}")
        sys.exit(1)
    
    print(f"[INFO] Agent kE shape: {agent_kE.shape}, kB shape: {agent_kB.shape}")
    print(f"[INFO] Standard kE shape: {std_kE.shape}, kB shape: {std_kB.shape}")
    
    # Get input data (g1_obs, g2_obs)
    g1_obs = args[0] if len(args) > 0 else kwargs.get('g1_obs')
    g2_obs = args[1] if len(args) > 1 else kwargs.get('g2_obs')
    
    if g1_obs is None or g2_obs is None:
        print("[ERROR] Could not extract g1_obs and g2_obs from inputs")
        sys.exit(1)
    
    # For evaluation, we need kappa_true, g1_true, g2_true, params
    # These are typically not available in the inversion function inputs
    # We'll create synthetic/dummy values for evaluation or compare outputs directly
    
    # Create results directories
    results_dir_agent = './results_agent'
    results_dir_std = './results_std'
    
    # Since we don't have ground truth (kappa_true), we'll use the standard output as reference
    # and compute metrics comparing agent output to standard output
    
    # Create dummy params based on input shape
    ny, nx = g1_obs.shape
    params = {
        'nx': nx,
        'ny': ny,
        'pixel_size': 1.0  # Assuming 1 arcmin per pixel
    }
    
    # For proper evaluation, we'll treat std_kE as "ground truth" for comparison
    # This tests whether the agent function produces equivalent results
    
    print("\n[EVAL] Comparing Agent output to Standard output...")
    
    # Direct comparison metrics
    kE_diff = agent_kE - std_kE
    kB_diff = agent_kB - std_kB
    
    kE_mse = np.mean(kE_diff**2)
    kB_mse = np.mean(kB_diff**2)
    
    kE_rmse = np.sqrt(kE_mse)
    kB_rmse = np.sqrt(kB_mse)
    
    # Correlation coefficients
    kE_cc = np.corrcoef(agent_kE.flatten(), std_kE.flatten())[0, 1]
    kB_cc = np.corrcoef(agent_kB.flatten(), std_kB.flatten())[0, 1]
    
    # Relative errors
    std_kE_rms = np.sqrt(np.mean(std_kE**2))
    std_kB_rms = np.sqrt(np.mean(std_kB**2))
    
    kE_rel_error = kE_rmse / std_kE_rms if std_kE_rms > 0 else float('inf')
    kB_rel_error = kB_rmse / std_kB_rms if std_kB_rms > 0 else float('inf')
    
    print(f"\n[COMPARISON] kE Metrics:")
    print(f"  RMSE: {kE_rmse:.6e}")
    print(f"  Correlation: {kE_cc:.6f}")
    print(f"  Relative Error: {kE_rel_error:.6e}")
    
    print(f"\n[COMPARISON] kB Metrics:")
    print(f"  RMSE: {kB_rmse:.6e}")
    print(f"  Correlation: {kB_cc:.6f}")
    print(f"  Relative Error: {kB_rel_error:.6e}")
    
    # Additional statistics
    print(f"\n[STATS] Agent kE range: [{agent_kE.min():.6f}, {agent_kE.max():.6f}]")
    print(f"[STATS] Standard kE range: [{std_kE.min():.6f}, {std_kE.max():.6f}]")
    print(f"[STATS] Agent kB range: [{agent_kB.min():.6f}, {agent_kB.max():.6f}]")
    print(f"[STATS] Standard kB range: [{std_kB.min():.6f}, {std_kB.max():.6f}]")
    
    # B/E ratio comparison
    agent_be_ratio = np.std(agent_kB) / np.std(agent_kE)
    std_be_ratio = np.std(std_kB) / np.std(std_kE)
    print(f"\n[STATS] Agent B/E ratio: {agent_be_ratio:.6f}")
    print(f"[STATS] Standard B/E ratio: {std_be_ratio:.6f}")
    
    # Determine success criteria
    # For numerical algorithms, we expect very high correlation (>0.99) 
    # and low relative error (<1e-6 for exact implementations)
    
    CORRELATION_THRESHOLD = 0.99
    RELATIVE_ERROR_THRESHOLD = 0.1  # 10% tolerance for numerical differences
    
    success = True
    
    if kE_cc < CORRELATION_THRESHOLD:
        print(f"\n[WARN] kE correlation ({kE_cc:.6f}) below threshold ({CORRELATION_THRESHOLD})")
        success = False
    
    if kB_cc < CORRELATION_THRESHOLD:
        print(f"\n[WARN] kB correlation ({kB_cc:.6f}) below threshold ({CORRELATION_THRESHOLD})")
        # B-mode is often noise-like, so we're more lenient
        if kB_cc < 0.9:
            success = False
    
    if kE_rel_error > RELATIVE_ERROR_THRESHOLD:
        print(f"\n[WARN] kE relative error ({kE_rel_error:.6f}) above threshold ({RELATIVE_ERROR_THRESHOLD})")
        success = False
    
    # Final verdict
    print("\n" + "="*60)
    if success:
        print("[RESULT] TEST PASSED - Agent output matches standard output")
        print(f"  kE Correlation: {kE_cc:.6f} (>= {CORRELATION_THRESHOLD})")
        print(f"  kE Relative Error: {kE_rel_error:.6e} (<= {RELATIVE_ERROR_THRESHOLD})")
        sys.exit(0)
    else:
        print("[RESULT] TEST FAILED - Agent output differs significantly from standard")
        print(f"  kE Correlation: {kE_cc:.6f} (threshold: {CORRELATION_THRESHOLD})")
        print(f"  kE Relative Error: {kE_rel_error:.6e} (threshold: {RELATIVE_ERROR_THRESHOLD})")
        sys.exit(1)


if __name__ == "__main__":
    main()