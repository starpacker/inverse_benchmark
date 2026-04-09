import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(stress_gt_vec, stress_rec_vec, stress_gt_2d, 
                     disp_clean, disp_noisy, nx, ny, 
                     results_dir, working_dir):
    """
    Compute metrics, visualize and save results.
    
    Parameters
    ----------
    stress_gt_vec : ndarray (n,)   Ground truth stress vector [MPa].
    stress_rec_vec : ndarray (n,)  Reconstructed stress vector [MPa].
    stress_gt_2d : ndarray (nx, ny)  Ground truth stress 2D [MPa].
    disp_clean : ndarray (n,)      Clean displacement [mm].
    disp_noisy : ndarray (n,)      Noisy displacement [mm].
    nx, ny : int                   Grid dimensions.
    results_dir : str              Directory to save results.
    working_dir : str              Working directory.
    
    Returns
    -------
    metrics : dict  Dictionary containing PSNR, SSIM, CC, RE, RMSE.
    """
    # Reshape for metrics computation
    gt = stress_gt_vec.reshape(nx, ny)
    rec = stress_rec_vec.reshape(nx, ny)
    
    # Compute metrics
    dr = gt.max() - gt.min()
    mse = np.mean((gt - rec)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt, rec, data_range=dr))
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), stress_gt_2d)
    
    # Also save to sandbox root for evaluation
    np.save(os.path.join(working_dir, "gt_output.npy"), stress_gt_2d)
    np.save(os.path.join(working_dir, "recon_output.npy"), rec)
    
    # Visualization
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    vmax = max(np.abs(stress_gt_2d).max(), np.abs(rec).max())
    
    im = axes[0, 0].imshow(stress_gt_2d.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='auto')
    axes[0, 0].set_title('(a) GT Residual Stress [MPa]')
    plt.colorbar(im, ax=axes[0, 0])
    
    im = axes[0, 1].imshow(rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='auto')
    axes[0, 1].set_title('(b) Reconstructed Stress')
    plt.colorbar(im, ax=axes[0, 1])
    
    err = stress_gt_2d - rec
    im = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 0].set_title('(c) Error')
    plt.colorbar(im, ax=axes[1, 0])
    
    axes[1, 1].plot(stress_gt_2d[:, ny//2], 'b-', lw=2, label='GT')
    axes[1, 1].plot(rec[:, ny//2], 'r--', lw=2, label='Recon')
    axes[1, 1].set_xlabel('x position')
    axes[1, 1].set_ylabel('Stress [MPa]')
    axes[1, 1].set_title('(d) Mid-depth Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f"pyCM — Residual Stress Contour Method\n"
                 f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
                 f"CC={metrics['CC']:.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics
