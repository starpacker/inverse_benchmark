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

def evaluate_results(dm_gt, dm_rec, stations, config, save_dir):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Args:
        dm_gt: ground truth velocity perturbation (nx, ny)
        dm_rec: reconstructed velocity perturbation (nx, ny)
        stations: station coordinates (n_stations, 2)
        config: dict with lat_min, lat_max, lon_min, lon_max
        save_dir: directory to save results
    
    Returns:
        dict containing metrics: PSNR, SSIM, CC, RE, RMSE
    """
    # Compute metrics
    gt_2d = dm_gt.copy()
    rec_2d = dm_rec.copy()
    data_range = gt_2d.max() - gt_2d.min()
    if data_range < 1e-12:
        data_range = 1.0
    
    mse = np.mean((gt_2d - rec_2d)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_2d, rec_2d, data_range=data_range))
    cc = float(np.corrcoef(gt_2d.ravel(), rec_2d.ravel())[0, 1])
    re = float(np.linalg.norm(gt_2d - rec_2d) / max(np.linalg.norm(gt_2d), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Visualization
    lat_min = config['lat_min']
    lat_max = config['lat_max']
    lon_min = config['lon_min']
    lon_max = config['lon_max']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmax = max(np.abs(dm_gt).max(), np.abs(dm_rec).max())
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    im0 = axes[0, 0].imshow(dm_gt.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='auto')
    axes[0, 0].plot(stations[:, 1], stations[:, 0], 'k^', ms=4)
    axes[0, 0].set_title('Ground Truth δc/c₀')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(dm_rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='auto')
    axes[0, 1].plot(stations[:, 1], stations[:, 0], 'k^', ms=4)
    axes[0, 1].set_title('LSQR Reconstruction')
    plt.colorbar(im1, ax=axes[0, 1])
    
    err = dm_gt - dm_rec
    im2 = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent, aspect='auto')
    axes[1, 0].set_title('Error (GT - Recon)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    mid = dm_gt.shape[0] // 2
    axes[1, 1].plot(dm_gt[mid, :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(dm_rec[mid, :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title(f'Cross-section (row {mid})')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Column index')
    axes[1, 1].set_ylabel('δc/c₀')
    
    fig.suptitle(
        f"seislib — Surface-Wave Tomography\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save data
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(save_dir, "reconstruction.npy"), dm_rec)
    np.save(os.path.join(save_dir, "ground_truth.npy"), dm_gt)
    
    return metrics
