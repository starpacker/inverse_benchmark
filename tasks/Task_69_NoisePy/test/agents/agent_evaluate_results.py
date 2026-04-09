import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(data, inversion_results, results_dir):
    """
    Compute evaluation metrics, save results, and generate visualizations.
    
    Metrics:
        - PSNR: Peak Signal-to-Noise Ratio
        - SSIM: Structural Similarity Index
        - CC: Correlation Coefficient
        - RE: Relative Error
        - RMSE: Root Mean Square Error
    
    Args:
        data: dict containing dm_gt, stations, pairs, and grid info
        inversion_results: dict containing best_rec and inversion info
        results_dir: directory to save outputs
        
    Returns:
        dict with computed metrics
    """
    dm_gt = data['dm_gt']
    stations = data['stations']
    pairs = data['pairs']
    xmin = data['xmin']
    xmax = data['xmax']
    ymin = data['ymin']
    ymax = data['ymax']
    
    best_rec = inversion_results['best_rec']
    best_alpha = inversion_results['best_alpha']
    
    # Compute metrics
    data_range = dm_gt.max() - dm_gt.min()
    if data_range < 1e-12:
        data_range = 1.0
    
    mse = np.mean((dm_gt - best_rec)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(dm_gt, best_rec, data_range=data_range))
    cc = float(np.corrcoef(dm_gt.ravel(), best_rec.ravel())[0, 1])
    re = float(np.linalg.norm(dm_gt - best_rec) / max(np.linalg.norm(dm_gt), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse
    }
    
    # Save metrics and arrays
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), dm_gt)
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmax = max(np.abs(dm_gt).max(), np.abs(best_rec).max())
    extent = [xmin, xmax, ymin, ymax]
    
    # Ray coverage
    ax = axes[0, 0]
    for si, sj in pairs[:200]:  # Plot subset of rays
        ax.plot([stations[si, 0], stations[sj, 0]],
                [stations[si, 1], stations[sj, 1]],
                'b-', alpha=0.05, lw=0.5)
    ax.plot(stations[:, 0], stations[:, 1], 'r^', ms=5)
    ax.set_title(f'Ray Coverage ({len(pairs)} paths)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    
    # Ground truth
    im1 = axes[0, 1].imshow(dm_gt.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             origin='lower', extent=extent, aspect='equal')
    axes[0, 1].set_title('Ground Truth δc/c₀')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Reconstruction
    im2 = axes[1, 0].imshow(best_rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             origin='lower', extent=extent, aspect='equal')
    axes[1, 0].set_title('LSQR Reconstruction')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Error
    err = dm_gt - best_rec
    im3 = axes[1, 1].imshow(err.T, cmap='RdBu_r', origin='lower',
                             extent=extent, aspect='equal')
    axes[1, 1].set_title('Error')
    plt.colorbar(im3, ax=axes[1, 1])
    
    fig.suptitle(
        f"NoisePy — Ambient Noise Tomography\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f} | "
        f"RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
