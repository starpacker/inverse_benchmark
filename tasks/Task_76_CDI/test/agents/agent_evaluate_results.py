import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

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
