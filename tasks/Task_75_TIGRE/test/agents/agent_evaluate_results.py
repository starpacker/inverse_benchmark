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

def compute_metrics(gt, rec):
    """
    Compute image quality metrics between ground truth and reconstruction.
    """
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}

def visualize_results(gt, sinogram, rec_fbp, rec_sirt, angles, metrics, save_path):
    """
    Create visualization of reconstruction results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(gt, cmap='gray')
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)')

    axes[0, 1].imshow(sinogram, aspect='auto', cmap='gray')
    axes[0, 1].set_title(f'Sinogram ({len(angles)} angles)')
    axes[0, 1].set_xlabel('Detector')
    axes[0, 1].set_ylabel('Angle index')

    axes[0, 2].imshow(rec_fbp / max(rec_fbp.max(), 1e-12), cmap='gray')
    axes[0, 2].set_title('FBP Reconstruction')

    axes[1, 0].imshow(rec_sirt / max(rec_sirt.max(), 1e-12), cmap='gray')
    axes[1, 0].set_title('SIRT Reconstruction')

    err = np.abs(gt / max(gt.max(), 1e-12) - rec_sirt / max(rec_sirt.max(), 1e-12))
    axes[1, 1].imshow(err, cmap='hot')
    axes[1, 1].set_title('|Error| (SIRT)')

    # Profile comparison
    mid = gt.shape[0] // 2
    axes[1, 2].plot(gt[mid, :] / max(gt[mid, :].max(), 1e-12), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_fbp[mid, :] / max(rec_fbp[mid, :].max(), 1e-12),
                     'g--', lw=1.5, label='FBP')
    axes[1, 2].plot(rec_sirt[mid, :] / max(rec_sirt[mid, :].max(), 1e-12),
                     'r--', lw=1.5, label='SIRT')
    axes[1, 2].set_title('Central Profile')
    axes[1, 2].legend()

    n_angles_sparse = len(angles)
    fig.suptitle(
        f"TIGRE — Sparse-View CT Reconstruction ({n_angles_sparse} views)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_results(phantom, rec_fbp, rec_sirt, sinogram_noisy, angles_sparse, results_dir, working_dir):
    """
    Evaluate reconstruction results, compute metrics, save outputs and visualizations.
    
    Parameters:
    -----------
    phantom : ndarray
        Ground truth phantom image
    rec_fbp : ndarray
        FBP reconstruction result
    rec_sirt : ndarray
        SIRT reconstruction result
    sinogram_noisy : ndarray
        Noisy sinogram data
    angles_sparse : ndarray
        Projection angles in degrees
    results_dir : str
        Directory to save results
    working_dir : str
        Working directory for additional outputs
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics for the best reconstruction
    best_reconstruction : ndarray
        The reconstruction with highest correlation coefficient
    method_name : str
        Name of the best method ('FBP' or 'SIRT')
    """
    # Compute metrics for both methods
    m_fbp = compute_metrics(phantom, rec_fbp)
    m_sirt = compute_metrics(phantom, rec_sirt)
    
    print("\n[EVALUATION] Metrics Comparison:")
    print(f"  FBP:  CC={m_fbp['CC']:.4f}, PSNR={m_fbp['PSNR']:.1f} dB, SSIM={m_fbp['SSIM']:.4f}")
    print(f"  SIRT: CC={m_sirt['CC']:.4f}, PSNR={m_sirt['PSNR']:.1f} dB, SSIM={m_sirt['SSIM']:.4f}")
    
    # Choose best reconstruction based on correlation coefficient
    if m_sirt['CC'] >= m_fbp['CC']:
        rec_best = rec_sirt
        metrics = m_sirt
        method = "SIRT"
    else:
        rec_best = rec_fbp
        metrics = m_fbp
        method = "FBP"
    
    print(f"\n  → Best method: {method} (higher CC)")
    
    # Print detailed metrics
    print("\n[FINAL METRICS]:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_best)
    np.save(os.path.join(results_dir, "ground_truth.npy"), phantom)
    
    # Also save to working dir for website assets
    np.save(os.path.join(working_dir, "gt_output.npy"), phantom)
    np.save(os.path.join(working_dir, "recon_output.npy"), rec_best)
    
    # Visualization
    visualize_results(phantom, sinogram_noisy, rec_fbp, rec_sirt,
                      angles_sparse, metrics,
                      os.path.join(results_dir, "reconstruction_result.png"))
    
    return metrics, rec_best, method
