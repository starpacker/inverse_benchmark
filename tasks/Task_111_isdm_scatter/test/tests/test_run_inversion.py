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

# ============================================================
# Inject the Referee (Evaluation Logic) from Reference B
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_111_isdm_scatter"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def align_and_compare(gt, recon):
    """
    Phase retrieval has ambiguities (translation, inversion).
    Try all 4 flips and pick best correlation.
    """
    best_cc = -1
    best_recon = recon.copy()

    candidates = [
        recon,
        np.flipud(recon),
        np.fliplr(recon),
        np.flipud(np.fliplr(recon)),
    ]

    for cand in candidates:
        # Try all circular shifts to find best alignment
        F_gt = np.fft.fft2(gt)
        F_cand = np.fft.fft2(cand)
        cross_corr = np.real(np.fft.ifft2(F_gt * np.conj(F_cand)))
        shift = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)

        aligned = np.roll(np.roll(cand, shift[0], axis=0), shift[1], axis=1)

        # Compute CC
        gt_norm = gt - np.mean(gt)
        al_norm = aligned - np.mean(aligned)
        denom = np.sqrt(np.sum(gt_norm**2) * np.sum(al_norm**2))
        if denom > 0:
            cc = np.sum(gt_norm * al_norm) / denom
        else:
            cc = 0

        if cc > best_cc:
            best_cc = cc
            best_recon = aligned.copy()

    return best_recon, best_cc


def evaluate_results(gt, recon_raw, speckle):
    """
    Align reconstruction, compute metrics, save outputs, and create visualization.
    
    Args:
        gt: Ground truth object
        recon_raw: Raw reconstruction from phase retrieval
        speckle: Speckle pattern for visualization
    
    Returns:
        metrics: Dictionary containing PSNR, SSIM, CC
        recon_aligned: Aligned reconstruction
    """
    # Align reconstruction (handle ambiguities)
    recon_aligned, _ = align_and_compare(gt, recon_raw)
    
    # Normalize both to [0, 1]
    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon_aligned - recon_aligned.min()) / (recon_aligned.max() - recon_aligned.min() + 1e-12)

    # PSNR
    mse = np.mean((gt_n - re_n)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    # SSIM
    ssim_val = ssim(gt_n, re_n, data_range=1.0)

    # CC
    g = gt_n - np.mean(gt_n)
    r = re_n - np.mean(re_n)
    denom = np.sqrt(np.sum(g**2) * np.sum(r**2))
    cc = np.sum(g * r) / (denom + 1e-12)

    metrics = {"PSNR": float(psnr), "SSIM": float(ssim_val), "CC": float(cc)}
    
    # Save outputs
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), recon_aligned)

    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt)
        np.save(os.path.join(d, "recon_output.npy"), recon_aligned)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    im0 = axes[0, 0].imshow(gt_n, cmap="gray")
    axes[0, 0].set_title("Ground Truth Object", fontsize=14)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.log1p(speckle), cmap="hot")
    axes[0, 1].set_title("Speckle Pattern (log scale)", fontsize=14)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(re_n, cmap="gray")
    axes[1, 0].set_title(
        f"HIO Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"SSIM={metrics['SSIM']:.4f}, CC={metrics['CC']:.4f}",
        fontsize=12,
    )
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    error = np.abs(gt_n - re_n)
    im3 = axes[1, 1].imshow(error, cmap="magma")
    axes[1, 1].set_title(f"Absolute Error (RMSE={np.sqrt(np.mean(error**2)):.4f})", fontsize=12)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics, recon_aligned


# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/isdm_scatter_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}, kwargs keys: {list(kwargs.keys())}")
    
    # Check for chained execution
    if inner_paths:
        # Pattern 2: Chained Execution
        print("Detected chained execution pattern.")
        print("Running outer function to get operator...")
        
        # Fix seed for reproducibility
        np.random.seed(42)
        agent_operator = run_inversion(*args, **kwargs)
        
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            np.random.seed(42)
            final_result = agent_operator(*inner_args, **inner_kwargs)
            
            # Evaluate
            # We need gt and speckle for evaluate_results
            # These would need to be extracted from context
            print("Chained execution completed.")
    else:
        # Pattern 1: Direct Execution
        print("Detected direct execution pattern.")
        
        # Fix seed for reproducibility
        np.random.seed(42)
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        final_result = agent_output
        std_result = std_output
    
    # Now evaluate both results
    # evaluate_results needs (gt, recon_raw, speckle)
    # We need to find gt and speckle. They are not direct inputs to run_inversion.
    # run_inversion takes: measured_magnitude, support, n_iter, beta, n_restarts
    # measured_magnitude is args[0], support is args[1]
    
    # For evaluation, we need the ground truth object.
    # Since we don't have gt directly, we use align_and_compare between std and agent results
    # as a proxy, or we compute metrics directly.
    
    # Actually, let's check if there's additional data we can use.
    # The evaluate_results function needs gt. Let's see if it's stored somewhere.
    # If not available, we compare agent vs standard using correlation-based metrics.
    
    print("\n=== Computing Quality Metrics ===")
    
    # Try to load ground truth if available
    gt = None
    gt_paths = [
        os.path.join(RESULTS_DIR, "gt_output.npy"),
        os.path.join(ASSETS_DIR, "gt_output.npy"),
    ]
    for gp in gt_paths:
        if os.path.exists(gp):
            gt = np.load(gp)
            print(f"Loaded ground truth from: {gp}")
            break
    
    # If gt is not available from files, we'll compare agent vs standard directly
    measured_magnitude = args[0] if len(args) > 0 else kwargs.get('measured_magnitude', None)
    
    # Use std_result as ground truth reference for comparison
    # Since phase retrieval is stochastic, we compare both against measured data
    
    # Compute R-factor for both
    def compute_r_factor(measured_magnitude, recon):
        F_recon = np.fft.fft2(recon)
        recon_mag = np.abs(F_recon)
        r_factor = np.sum(np.abs(measured_magnitude - recon_mag)) / np.sum(measured_magnitude + 1e-12)
        return r_factor
    
    if final_result is not None and measured_magnitude is not None:
        r_agent = compute_r_factor(measured_magnitude, final_result)
        print(f"Agent R-factor: {r_agent:.6f}")
    else:
        r_agent = float('inf')
        print("WARNING: Could not compute agent R-factor")
    
    if std_result is not None and measured_magnitude is not None:
        r_std = compute_r_factor(measured_magnitude, std_result)
        print(f"Standard R-factor: {r_std:.6f}")
    else:
        r_std = float('inf')
        print("WARNING: Could not compute standard R-factor")
    
    # Also compare agent output to standard output using cross-correlation
    if final_result is not None and std_result is not None:
        # Align agent to standard
        aligned_agent, cc_val = align_and_compare(std_result, final_result)
        print(f"Cross-correlation (agent vs standard): {cc_val:.6f}")
        
        # If gt is available, run full evaluate_results
        if gt is not None:
            speckle = measured_magnitude**2 if measured_magnitude is not None else np.zeros_like(gt)
            
            print("\n--- Agent Evaluation ---")
            metrics_agent, _ = evaluate_results(gt, final_result, speckle)
            print(f"Agent metrics: PSNR={metrics_agent['PSNR']:.2f}, SSIM={metrics_agent['SSIM']:.4f}, CC={metrics_agent['CC']:.4f}")
            
            print("\n--- Standard Evaluation ---")
            metrics_std, _ = evaluate_results(gt, std_result, speckle)
            print(f"Standard metrics: PSNR={metrics_std['PSNR']:.2f}, SSIM={metrics_std['SSIM']:.4f}, CC={metrics_std['CC']:.4f}")
            
            score_agent = metrics_agent['CC']
            score_std = metrics_std['CC']
        else:
            # No GT available; use R-factor (lower is better) and CC against standard
            score_agent = cc_val
            score_std = 1.0  # perfect self-correlation
            
            # For R-factor comparison
            print(f"\nR-factor comparison: Agent={r_agent:.6f}, Standard={r_std:.6f}")
    else:
        print("ERROR: Missing results for comparison")
        sys.exit(1)
    
    # ============================================================
    # Verification & Reporting
    # ============================================================
    print("\n=== Final Verification ===")
    print(f"Scores -> Agent: {score_agent:.6f}, Standard: {score_std:.6f}")
    
    # For R-factor: lower is better
    # For CC/PSNR/SSIM: higher is better
    
    # Primary check: R-factor should not be dramatically worse
    # R-factor is a "lower is better" metric
    r_factor_ok = True
    if r_agent != float('inf') and r_std != float('inf'):
        # Allow agent R-factor to be up to 20% worse than standard
        if r_agent > r_std * 1.20:
            print(f"WARNING: Agent R-factor ({r_agent:.6f}) is >20% worse than standard ({r_std:.6f})")
            r_factor_ok = False
        else:
            print(f"R-factor check PASSED: Agent={r_agent:.6f} vs Standard={r_std:.6f}")
    
    # Secondary check: CC/quality metric
    # score_agent should be at least 90% of score_std for CC-based metrics
    # or within acceptable range
    cc_ok = True
    if gt is not None:
        # We have full metrics - use CC as primary
        margin = 0.10  # 10% margin
        threshold = score_std * (1.0 - margin)
        if score_agent < threshold:
            print(f"WARNING: Agent CC ({score_agent:.6f}) below threshold ({threshold:.6f})")
            cc_ok = False
        else:
            print(f"CC check PASSED: Agent={score_agent:.6f} >= threshold={threshold:.6f}")
    else:
        # Compare CC against standard output
        # Agent should have reasonable correlation with standard
        if score_agent < 0.5:
            print(f"WARNING: Low correlation with standard ({score_agent:.6f})")
            cc_ok = False
        else:
            print(f"Correlation check PASSED: CC={score_agent:.6f}")
    
    # Final decision - be lenient since phase retrieval is stochastic
    # R-factor is the most reliable metric here
    if r_factor_ok or cc_ok:
        print("\n*** TEST PASSED ***")
        sys.exit(0)
    else:
        print("\n*** TEST FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)