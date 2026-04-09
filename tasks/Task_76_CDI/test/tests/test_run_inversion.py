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


def evaluate_results(obj_gt, obj_rec, intensity_noisy, errors, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        obj_gt: Ground truth complex object
        obj_rec: Reconstructed complex object
        intensity_noisy: Noisy diffraction intensity
        errors: Convergence history
        results_dir: Directory to save results
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Phase alignment: remove global phase ambiguity and possible twin-image flip
    candidates = [obj_rec, np.conj(obj_rec), np.flip(obj_rec),
                  np.conj(np.flip(obj_rec))]
    
    best_cc = -1
    best = obj_rec
    
    for cand in candidates:
        # Find optimal global phase
        cross = np.sum(obj_gt * np.conj(cand))
        phi = np.angle(cross)
        cand_aligned = cand * np.exp(1j * phi)
        
        cc = np.abs(np.corrcoef(
            np.abs(obj_gt).ravel(), np.abs(cand_aligned).ravel()
        )[0, 1])
        if cc > best_cc:
            best_cc = cc
            best = cand_aligned
    
    obj_rec_aligned = best
    
    # Compute metrics
    amp_gt = np.abs(obj_gt)
    amp_rec = np.abs(obj_rec_aligned)
    
    amp_gt_n = amp_gt / max(amp_gt.max(), 1e-12)
    amp_rec_n = amp_rec / max(amp_rec.max(), 1e-12)
    
    data_range = 1.0
    mse = np.mean((amp_gt_n - amp_rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(amp_gt_n, amp_rec_n, data_range=data_range))
    cc = float(np.corrcoef(amp_gt_n.ravel(), amp_rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(amp_gt_n - amp_rec_n) /
               max(np.linalg.norm(amp_gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    
    # Phase error (inside support only)
    support = amp_gt > 0.01 * amp_gt.max()
    if support.sum() > 0:
        phase_gt = np.angle(obj_gt[support])
        phase_rec = np.angle(obj_rec_aligned[support])
        phase_err = np.angle(np.exp(1j * (phase_gt - phase_rec)))
        phase_rmse = float(np.sqrt(np.mean(phase_err**2)))
    else:
        phase_rmse = np.pi
    
    metrics = {
        "PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse,
        "phase_RMSE_rad": phase_rmse,
    }
    
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    
    std_metrics = {k: v for k, v in metrics.items()
                   if k in ["PSNR", "SSIM", "CC", "RE", "RMSE"]}
    std_metrics["phase_RMSE_rad"] = metrics["phase_RMSE_rad"]
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(std_metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), np.abs(obj_rec_aligned))
    np.save(os.path.join(results_dir, "ground_truth.npy"), np.abs(obj_gt))
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    
    # Diffraction pattern
    axes[0, 0].imshow(np.log10(intensity_noisy + 1e-6), cmap='viridis')
    axes[0, 0].set_title('Diffraction Pattern (log)')
    
    # GT amplitude
    axes[0, 1].imshow(np.abs(obj_gt), cmap='gray')
    axes[0, 1].set_title('GT Amplitude')
    
    # Recon amplitude
    axes[0, 2].imshow(np.abs(obj_rec_aligned), cmap='gray')
    axes[0, 2].set_title('Recon Amplitude')
    
    # Amplitude error
    err_amp = np.abs(np.abs(obj_gt) - np.abs(obj_rec_aligned))
    axes[0, 3].imshow(err_amp, cmap='hot')
    axes[0, 3].set_title('|Amplitude Error|')
    
    # GT phase
    phase_gt_vis = np.angle(obj_gt) * support
    axes[1, 0].imshow(phase_gt_vis, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('GT Phase')
    
    # Recon phase
    phase_rec_vis = np.angle(obj_rec_aligned) * support
    axes[1, 1].imshow(phase_rec_vis, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Recon Phase')
    
    # Phase error
    phase_err_vis = np.angle(np.exp(1j * (phase_gt_vis - phase_rec_vis))) * support
    axes[1, 2].imshow(phase_err_vis, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('Phase Error')
    
    # Convergence
    if errors:
        axes[1, 3].semilogy(errors)
        axes[1, 3].set_title('Convergence (R-factor)')
        axes[1, 3].set_xlabel('Iteration')
        axes[1, 3].grid(True)
    
    for row in axes:
        for ax in row:
            if ax != axes[1, 3]:
                ax.axis('off')
    
    fig.suptitle(
        f"CDI — Phase Retrieval (HIO+ER)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f} | "
        f"Phase RMSE={metrics['phase_RMSE_rad']:.3f} rad",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/CDI_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data paths
    outer_paths = []
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_paths.append(p)
    
    print(f"Outer data files: {outer_paths}")
    print(f"Inner data files: {inner_paths}")
    
    try:
        # Load outer (primary) data
        if not outer_paths:
            print("ERROR: No primary data file found!")
            sys.exit(1)
        
        outer_path = outer_paths[0]
        print(f"\nLoading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Function: {outer_data.get('func_name', 'unknown')}")
        print(f"Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
        
        # Execute run_inversion
        print("\n[STAGE 1] Running agent run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_paths:
            print("\n[STAGE 2] Chained execution detected...")
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned callable
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                print("WARNING: Agent output is not callable, using directly")
                final_result = agent_output
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # The function returns (obj_rec, support_final, errors)
        # We need ground truth for evaluation - extract from input args
        # Based on the function signature: intensity_noisy is args[0], support_init is args[1]
        intensity_noisy = args[0]
        
        # Extract agent results
        if isinstance(final_result, tuple) and len(final_result) >= 3:
            agent_obj_rec, agent_support, agent_errors = final_result[0], final_result[1], final_result[2]
        else:
            print("ERROR: Unexpected result format from agent")
            sys.exit(1)
        
        # Extract standard results
        if isinstance(std_result, tuple) and len(std_result) >= 3:
            std_obj_rec, std_support, std_errors = std_result[0], std_result[1], std_result[2]
        else:
            print("ERROR: Unexpected result format from standard")
            sys.exit(1)
        
        # For evaluation, we need obj_gt (ground truth complex object)
        # Since we don't have ground truth in the data, we use std_obj_rec as reference
        # This tests if agent produces similar quality to standard implementation
        obj_gt = std_obj_rec
        
        print("\n[STAGE 3] Evaluating Agent Results...")
        results_dir_agent = "./results_agent"
        metrics_agent = evaluate_results(obj_gt, agent_obj_rec, intensity_noisy, agent_errors, results_dir_agent)
        
        print("\n[STAGE 3] Evaluating Standard Results...")
        results_dir_std = "./results_std"
        metrics_std = evaluate_results(obj_gt, std_obj_rec, intensity_noisy, std_errors, results_dir_std)
        
        # Compare metrics - use PSNR as primary metric (higher is better)
        score_agent = metrics_agent['PSNR']
        score_std = metrics_std['PSNR']
        
        print(f"\n[STAGE 5] Comparison:")
        print(f"  Agent PSNR: {score_agent:.4f}")
        print(f"  Standard PSNR: {score_std:.4f}")
        print(f"  Agent CC: {metrics_agent['CC']:.4f}")
        print(f"  Standard CC: {metrics_std['CC']:.4f}")
        print(f"  Agent SSIM: {metrics_agent['SSIM']:.4f}")
        print(f"  Standard SSIM: {metrics_std['SSIM']:.4f}")
        
        # Also compare final R-factor errors
        agent_final_err = agent_errors[-1] if agent_errors else float('inf')
        std_final_err = std_errors[-1] if std_errors else float('inf')
        print(f"  Agent final R-factor: {agent_final_err:.6f}")
        print(f"  Standard final R-factor: {std_final_err:.6f}")
        
        # Success criteria:
        # 1. PSNR should be at least 90% of standard (allowing 10% margin)
        # 2. CC (correlation coefficient) should be >= 0.9 of standard
        # 3. R-factor error should not be more than 20% worse than standard
        
        psnr_threshold = score_std * 0.90  # 10% margin
        cc_threshold = metrics_std['CC'] * 0.90
        
        # For R-factor, lower is better, so we check if agent is not too much worse
        rfactor_threshold = std_final_err * 1.20  # Allow 20% worse
        
        success = True
        reasons = []
        
        if score_agent < psnr_threshold:
            success = False
            reasons.append(f"PSNR too low: {score_agent:.4f} < {psnr_threshold:.4f}")
        
        if metrics_agent['CC'] < cc_threshold:
            success = False
            reasons.append(f"CC too low: {metrics_agent['CC']:.4f} < {cc_threshold:.4f}")
        
        if agent_final_err > rfactor_threshold:
            success = False
            reasons.append(f"R-factor too high: {agent_final_err:.6f} > {rfactor_threshold:.6f}")
        
        print(f"\nScores -> Agent PSNR: {score_agent:.4f}, Standard PSNR: {score_std:.4f}")
        
        if success:
            print("\n[RESULT] SUCCESS: Agent performance is acceptable!")
            sys.exit(0)
        else:
            print(f"\n[RESULT] FAILURE: Agent performance degraded!")
            for r in reasons:
                print(f"  - {r}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Exception occurred during testing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()