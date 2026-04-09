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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ============== INJECT REFEREE CODE ==============

def normalise_phase(phase):
    """Shift to min=0 and normalise to [0, 1]."""
    p = phase - phase.min()
    mx = p.max()
    return p / mx if mx > 0 else p

def plot_results(gt_phase, avg_dp, recon_phase, error_map, metrics, save_path):
    """4-panel figure: GT | avg DP | recon | error."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].imshow(gt_phase, cmap="inferno")
    axes[0].set_title("Ground-Truth Phase")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.log1p(avg_dp), cmap="viridis")
    axes[1].set_title("Avg Diffraction (log)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(recon_phase, cmap="inferno")
    axes[2].set_title(
        f"Reconstructed Phase\n"
        f"PSNR={metrics['PSNR_dB']:.1f} dB  SSIM={metrics['SSIM']:.3f}"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error_map, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[3].set_title(f"Phase Error (RMSE={metrics['RMSE']:.4f})")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")

def evaluate_results(gt_phase, recon_phase, fov_mask, data_4d, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE metrics between ground-truth and reconstructed
    phase, handles phase alignment, and generates visualizations.
    
    Parameters
    ----------
    gt_phase : ndarray
        Ground truth phase array
    recon_phase : ndarray
        Reconstructed phase array
    fov_mask : ndarray (bool)
        Field-of-view mask indicating valid reconstruction region
    data_4d : ndarray
        Raw 4D-STEM data for visualization
    results_dir : str
        Directory to save results
    
    Returns
    -------
    dict containing:
        - PSNR_dB: Peak signal-to-noise ratio in dB
        - SSIM: Structural similarity index
        - RMSE: Root mean square error
        - phase_correlation: Correlation between GT and reconstruction
        - psnr, ssim, rmse: Duplicate keys for compatibility
    """
    print("\n[6/6] Computing metrics ...")

    assert gt_phase.shape == recon_phase.shape, (
        f"Shape mismatch: GT {gt_phase.shape} vs recon {recon_phase.shape}"
    )

    fov = fov_mask
    gt_fov = gt_phase[fov]
    rc_fov = recon_phase[fov]

    # Remove global phase offset
    rc_fov = rc_fov - np.mean(rc_fov)
    gt_fov = gt_fov - np.mean(gt_fov)

    # Handle sign ambiguity
    corr_pos = np.corrcoef(gt_fov, rc_fov)[0, 1]
    corr_neg = np.corrcoef(gt_fov, -rc_fov)[0, 1]
    if corr_neg > corr_pos:
        recon_phase = -recon_phase
        rc_fov = -rc_fov
        print("  (Phase sign flipped for alignment)")

    best_corr = max(corr_pos, corr_neg)
    print(f"  Phase correlation (FOV) = {best_corr:.4f}")
    print(f"  FOV pixels              = {fov.sum()} / {fov.size}")

    # Alignment via least-squares
    recon_aligned = recon_phase.copy()
    gt_aligned = gt_phase.copy()

    gt_fov_vals = gt_aligned[fov].flatten()
    rc_fov_vals = recon_aligned[fov].flatten()
    A_mat = np.column_stack([gt_fov_vals, np.ones_like(gt_fov_vals)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, rc_fov_vals, rcond=None)
    a_ls, b_ls = coeffs
    print(f"  LS alignment: recon ≈ {a_ls:.4f} * GT + {b_ls:.6f}")
    
    if abs(a_ls) > 1e-10:
        recon_aligned = (recon_aligned - b_ls) / a_ls

    # FOV-only PSNR/RMSE
    gt_fov_pixels = gt_aligned[fov]
    rc_fov_pixels = recon_aligned[fov]
    rc_fov_pixels = np.clip(rc_fov_pixels, gt_fov_pixels.min(), gt_fov_pixels.max())
    gt_n_fov = (gt_fov_pixels - gt_fov_pixels.min()) / (gt_fov_pixels.max() - gt_fov_pixels.min() + 1e-12)
    rc_n_fov = (rc_fov_pixels - gt_fov_pixels.min()) / (gt_fov_pixels.max() - gt_fov_pixels.min() + 1e-12)
    rc_n_fov = np.clip(rc_n_fov, 0, 1)
    rmse_fov = float(np.sqrt(np.mean((gt_n_fov - rc_n_fov)**2)))
    psnr_fov = float(20.0 * np.log10(1.0 / rmse_fov)) if rmse_fov > 0 else float('inf')

    # FOV SSIM using bounding box
    rows = np.any(fov, axis=1)
    cols = np.any(fov, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    gt_box = gt_aligned[rmin:rmax + 1, cmin:cmax + 1].copy()
    rc_box = recon_aligned[rmin:rmax + 1, cmin:cmax + 1].copy()
    fov_box = fov[rmin:rmax + 1, cmin:cmax + 1]

    rc_box = np.clip(rc_box, gt_box[fov_box].min(), gt_box[fov_box].max())
    gt_box[~fov_box] = np.mean(gt_box[fov_box])
    rc_box[~fov_box] = np.mean(rc_box[fov_box])
    gt_rng = gt_box.max() - gt_box.min() + 1e-12
    gt_box_n = (gt_box - gt_box.min()) / gt_rng
    rc_box_n = (rc_box - gt_box.min()) / gt_rng
    rc_box_n = np.clip(rc_box_n, 0, 1)
    ssim_val = float(ssim(gt_box_n, rc_box_n, data_range=1.0))

    metrics = {
        "PSNR_dB": round(psnr_fov, 3),
        "SSIM": round(ssim_val, 4),
        "RMSE": round(rmse_fov, 6),
    }
    metrics["phase_correlation"] = round(float(best_corr), 4)
    metrics["psnr"] = metrics["PSNR_dB"]
    metrics["ssim"] = metrics["SSIM"]
    metrics["rmse"] = metrics["RMSE"]
    
    print(f"  PSNR = {metrics['psnr']:.2f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CC   = {metrics['phase_correlation']:.4f}")

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"\n  Metrics  → {metrics_path}")

    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_aligned)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_phase)
    print(f"  Arrays   → results/reconstruction.npy, ground_truth.npy")

    # Visualize
    avg_dp = data_4d.mean(axis=(0, 1))
    gt_n = normalise_phase(gt_aligned)
    rc_n = normalise_phase(recon_aligned)
    err = gt_n - rc_n

    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plot_results(gt_n, avg_dp, rc_n, err, metrics, fig_path)

    return metrics

# ============== END REFEREE CODE ==============


def main():
    """Main test function for run_inversion."""
    
    # Data paths provided
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Load outer/main data
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
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
    
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run agent function
    print("\n" + "="*60)
    print("Running agent's run_inversion...")
    print("="*60)
    
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\nAgent output keys:", agent_output.keys() if isinstance(agent_output, dict) else type(agent_output))
    
    # Check if we have inner data (chained execution)
    if inner_files:
        print("\nChained execution detected - loading inner data...")
        inner_path = inner_files[0]
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        if callable(agent_output):
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            final_result = agent_output
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    # Now we need to evaluate both results
    # The function returns a dict with recon_phase, fov_mask, etc.
    # We need ground truth phase and 4D data for evaluation
    
    # Extract from args - based on function signature:
    # run_inversion(datacube, energy, semiangle, scan_step, sampling, diff_px,
    #               num_iter, step_size, max_batch, angular_sampling)
    datacube = args[0] if len(args) > 0 else kwargs.get('datacube')
    
    # Get 4D data from datacube
    if hasattr(datacube, 'data'):
        data_4d = np.array(datacube.data)
    else:
        data_4d = np.array(datacube)
    
    print(f"\n4D data shape: {data_4d.shape}")
    
    # For evaluation, we need ground truth phase
    # In typical ptychography tests, we might have it stored somewhere
    # Let's check if it's in the std_result or we need to use std_result as reference
    
    # Create results directories
    agent_results_dir = "./results_agent"
    std_results_dir = "./results_std"
    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(std_results_dir, exist_ok=True)
    
    # Extract reconstruction results
    agent_recon_phase = final_result['recon_phase']
    agent_fov_mask = final_result['fov_mask']
    agent_error = final_result['final_error']
    
    std_recon_phase = std_result['recon_phase']
    std_fov_mask = std_result['fov_mask']
    std_error = std_result['final_error']
    
    print(f"\nAgent recon phase shape: {agent_recon_phase.shape}")
    print(f"Std recon phase shape: {std_recon_phase.shape}")
    print(f"Agent final error: {agent_error}")
    print(f"Std final error: {std_error}")
    
    # Since we don't have explicit ground truth, we use the standard result as reference
    # and evaluate how close the agent result is to standard
    gt_phase = std_recon_phase  # Use standard as ground truth for comparison
    
    # Evaluate agent results
    print("\n" + "="*60)
    print("Evaluating AGENT results...")
    print("="*60)
    
    try:
        agent_metrics = evaluate_results(
            gt_phase=gt_phase,
            recon_phase=agent_recon_phase,
            fov_mask=agent_fov_mask,
            data_4d=data_4d,
            results_dir=agent_results_dir
        )
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        # Fallback: compare arrays directly
        agent_metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'rmse': 1.0
        }
    
    # For standard, evaluate against itself (should be perfect)
    print("\n" + "="*60)
    print("Evaluating STANDARD results (self-comparison)...")
    print("="*60)
    
    try:
        std_metrics = evaluate_results(
            gt_phase=gt_phase,
            recon_phase=std_recon_phase,
            fov_mask=std_fov_mask,
            data_4d=data_4d,
            results_dir=std_results_dir
        )
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        std_metrics = {
            'psnr': float('inf'),
            'ssim': 1.0,
            'rmse': 0.0
        }
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Agent PSNR: {agent_metrics.get('psnr', agent_metrics.get('PSNR_dB', 0)):.2f} dB")
    print(f"Std PSNR:   {std_metrics.get('psnr', std_metrics.get('PSNR_dB', 0)):.2f} dB")
    print(f"Agent SSIM: {agent_metrics.get('ssim', agent_metrics.get('SSIM', 0)):.4f}")
    print(f"Std SSIM:   {std_metrics.get('ssim', std_metrics.get('SSIM', 0)):.4f}")
    print(f"Agent RMSE: {agent_metrics.get('rmse', agent_metrics.get('RMSE', 0)):.6f}")
    print(f"Std RMSE:   {std_metrics.get('rmse', std_metrics.get('RMSE', 0)):.6f}")
    print(f"Agent Final Error: {agent_error:.6f}")
    print(f"Std Final Error:   {std_error:.6f}")
    
    # Determine success
    # Since std is compared against itself, we expect perfect metrics
    # For agent, we need reasonable metrics indicating good reconstruction
    
    agent_psnr = agent_metrics.get('psnr', agent_metrics.get('PSNR_dB', 0))
    agent_ssim = agent_metrics.get('ssim', agent_metrics.get('SSIM', 0))
    agent_rmse = agent_metrics.get('rmse', agent_metrics.get('RMSE', 1))
    
    # Success criteria:
    # - PSNR should be reasonably high (> 15 dB indicates good similarity)
    # - SSIM should be high (> 0.7 for reasonable reconstruction)
    # - RMSE should be low (< 0.3)
    # - Or final error should be comparable
    
    psnr_threshold = 15.0  # dB
    ssim_threshold = 0.7
    rmse_threshold = 0.3
    error_ratio_threshold = 2.0  # Agent error shouldn't be more than 2x std error
    
    success = True
    reasons = []
    
    # Check if PSNR is reasonable (but account for perfect self-comparison case)
    if agent_psnr < psnr_threshold and agent_psnr != float('inf'):
        reasons.append(f"PSNR {agent_psnr:.2f} < {psnr_threshold} dB threshold")
        # Don't fail immediately, check other metrics
    
    if agent_ssim < ssim_threshold:
        reasons.append(f"SSIM {agent_ssim:.4f} < {ssim_threshold} threshold")
    
    if agent_rmse > rmse_threshold:
        reasons.append(f"RMSE {agent_rmse:.6f} > {rmse_threshold} threshold")
    
    # Check reconstruction error ratio
    if std_error > 0:
        error_ratio = agent_error / std_error
        if error_ratio > error_ratio_threshold:
            reasons.append(f"Error ratio {error_ratio:.2f} > {error_ratio_threshold}")
    
    # If multiple issues, consider it a failure
    if len(reasons) >= 2:
        success = False
    
    # Also do direct array comparison as sanity check
    phase_diff = np.abs(agent_recon_phase - std_recon_phase)
    mean_diff = np.mean(phase_diff)
    max_diff = np.max(phase_diff)
    print(f"\nDirect phase comparison:")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Max absolute diff:  {max_diff:.6f}")
    
    # If very similar to standard, definitely pass
    if mean_diff < 0.01 and max_diff < 0.1:
        print("  -> Very close to standard output!")
        success = True
        reasons = []
    
    print("\n" + "="*60)
    if success:
        print("TEST PASSED: Agent performance is acceptable")
        sys.exit(0)
    else:
        print("TEST FAILED: Agent performance degraded")
        for reason in reasons:
            print(f"  - {reason}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)