import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(data_dict, result_dict, results_dir):
    """
    Evaluate reconstruction quality and generate visualization.
    
    Parameters:
        data_dict: Dictionary containing ground truth and grid data
        result_dict: Dictionary containing reconstructed pressure
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary of quality metrics
    """
    p_gt = data_dict['p_gt']
    p_rec = result_dict['p_rec']
    xx = data_dict['xx']
    yy = data_dict['yy']
    u_noisy = data_dict['u_noisy']
    v_noisy = data_dict['v_noisy']
    
    # Compute metrics (mean-removed)
    p_gt_zm = p_gt - p_gt.mean()
    p_rec_zm = p_rec - p_rec.mean()
    data_range = p_gt_zm.max() - p_gt_zm.min()
    if data_range < 1e-12:
        data_range = 1.0
    
    mse = np.mean((p_gt_zm - p_rec_zm)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(p_gt_zm, p_rec_zm, data_range=data_range))
    cc = float(np.corrcoef(p_gt_zm.ravel(), p_rec_zm.ravel())[0, 1])
    re = float(np.linalg.norm(p_gt_zm - p_rec_zm) / max(np.linalg.norm(p_gt_zm), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse,
        "method_used": result_dict['method_used']
    }
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), p_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), p_gt)
    
    # Visualization
    vmax = max(np.abs(p_gt_zm).max(), np.abs(p_rec_zm).max())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Velocity magnitude
    speed = np.sqrt(u_noisy**2 + v_noisy**2)
    im0 = axes[0, 0].contourf(xx, yy, speed, levels=30, cmap='viridis')
    axes[0, 0].set_title('Velocity Magnitude |V|')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # GT pressure
    im1 = axes[0, 1].contourf(xx, yy, p_gt_zm, levels=30, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Ground Truth Pressure')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Reconstructed pressure
    im2 = axes[1, 0].contourf(xx, yy, p_rec_zm, levels=30, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Reconstructed Pressure')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Error
    err = p_gt_zm - p_rec_zm
    im3 = axes[1, 1].contourf(xx, yy, err, levels=30, cmap='RdBu_r')
    axes[1, 1].set_title('Error (GT - Recon)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    fig.suptitle(
        f"SPICY — Pressure from PIV Reconstruction\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
