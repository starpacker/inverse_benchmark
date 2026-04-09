import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(ground_truth, reconstruction, x_traces, z_depth, t_axis, bscan_noisy, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        ground_truth: true reflectivity model (nx, nz)
        reconstruction: migrated/reconstructed image (nx, nz)
        x_traces: trace positions array
        z_depth: depth axis array
        t_axis: time axis array
        bscan_noisy: noisy B-scan data for visualization
        results_dir: directory to save outputs
    
    Returns:
        metrics: dictionary containing PSNR, SSIM, CC, RE, RMSE
    """
    # Normalize for comparison
    gt_n = ground_truth / max(ground_truth.max(), 1e-12)
    rec_n = reconstruction / max(reconstruction.max(), 1e-12)
    data_range = 1.0
    
    # Compute metrics
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse
    }
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), reconstruction)
    np.save(os.path.join(results_dir, "ground_truth.npy"), ground_truth)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Subsurface model
    axes[0, 0].imshow(ground_truth.T, aspect='auto', cmap='gray_r',
                       extent=[x_traces[0], x_traces[-1], z_depth[-1], z_depth[0]])
    axes[0, 0].set_title('True Reflectivity Model')
    axes[0, 0].set_xlabel('Position [m]')
    axes[0, 0].set_ylabel('Depth [m]')
    
    # B-scan
    clip = np.percentile(np.abs(bscan_noisy), 98)
    axes[0, 1].imshow(bscan_noisy.T, aspect='auto', cmap='RdBu_r', vmin=-clip, vmax=clip,
                       extent=[x_traces[0], x_traces[-1], t_axis[-1]*1e9, t_axis[0]*1e9])
    axes[0, 1].set_title('GPR B-Scan (noisy)')
    axes[0, 1].set_xlabel('Position [m]')
    axes[0, 1].set_ylabel('Two-way time [ns]')
    
    # Migrated image
    axes[1, 0].imshow(reconstruction.T, aspect='auto', cmap='gray_r',
                       extent=[x_traces[0], x_traces[-1], z_depth[-1], z_depth[0]])
    axes[1, 0].set_title('Kirchhoff Migration')
    axes[1, 0].set_xlabel('Position [m]')
    axes[1, 0].set_ylabel('Depth [m]')
    
    # Cross-section comparison
    mid = ground_truth.shape[0] // 2
    axes[1, 1].plot(z_depth, ground_truth[mid, :] / max(ground_truth[mid, :].max(), 1e-12),
                     'b-', lw=2, label='GT')
    axes[1, 1].plot(z_depth, reconstruction[mid, :] / max(reconstruction[mid, :].max(), 1e-12),
                     'r--', lw=2, label='Migrated')
    axes[1, 1].set_title(f'Trace {mid} Comparison')
    axes[1, 1].set_xlabel('Depth [m]')
    axes[1, 1].legend()
    
    fig.suptitle(
        f"GPRPy — GPR Migration\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
