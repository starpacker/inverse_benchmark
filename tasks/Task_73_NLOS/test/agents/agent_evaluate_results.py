import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(rho_gt, rho_rec, transient, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Parameters:
        rho_gt: ndarray (nx, ny, nz) - Ground truth hidden scene
        rho_rec: ndarray (nx, ny, nz) - Reconstructed hidden scene
        transient: ndarray (nx, ny, n_time) - Transient measurements
        results_dir: str - Directory to save results
    
    Returns:
        metrics: dict containing PSNR, SSIM, CC, RE, RMSE
    """
    # Max intensity projections for comparison
    gt_mip = rho_gt.max(axis=2)
    rec_mip = rho_rec.max(axis=2)
    
    # Normalize GT to [0, 1]
    gt_n = gt_mip / max(gt_mip.max(), 1e-12)
    
    # Least-squares alignment of rec_mip to gt_mip
    rec_flat = rec_mip.ravel()
    gt_flat = gt_n.ravel()
    A_mat = np.column_stack([rec_flat, np.ones_like(rec_flat)])
    result = np.linalg.lstsq(A_mat, gt_flat, rcond=None)
    a, b = result[0]
    rec_n = np.clip(a * rec_mip + b, 0, 1)
    
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Save metrics and data
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), rho_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), rho_gt)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # GT MIP
    axes[0, 0].imshow(gt_mip, cmap='hot', origin='lower')
    axes[0, 0].set_title('GT — Max Intensity Projection (XY)')
    
    # Recon MIP
    axes[0, 1].imshow(rec_mip / max(rec_mip.max(), 1e-12), cmap='hot', origin='lower')
    axes[0, 1].set_title('Recon — MIP (XY)')
    
    # Transient slice
    mid_x = transient.shape[0] // 2
    axes[0, 2].imshow(transient[mid_x, :, :].T, aspect='auto', cmap='viridis',
                       origin='lower')
    axes[0, 2].set_title(f'Transient τ(x={mid_x}, y, t)')
    axes[0, 2].set_xlabel('y index')
    axes[0, 2].set_ylabel('Time bin')
    
    # GT depth slice
    gt_side = rho_gt.max(axis=1)
    axes[1, 0].imshow(gt_side.T, cmap='hot', origin='lower', aspect='auto')
    axes[1, 0].set_title('GT — MIP (XZ)')
    
    # Recon depth slice
    rec_side = rho_rec.max(axis=1)
    axes[1, 1].imshow(rec_side.T / max(rec_side.max(), 1e-12),
                       cmap='hot', origin='lower', aspect='auto')
    axes[1, 1].set_title('Recon — MIP (XZ)')
    
    # Error
    err = np.abs(gt_mip - rec_mip / max(rec_mip.max(), 1e-12))
    axes[1, 2].imshow(err, cmap='hot', origin='lower')
    axes[1, 2].set_title('|Error| (XY)')
    
    fig.suptitle(
        f"NLOS — Non-Line-of-Sight Reconstruction\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
