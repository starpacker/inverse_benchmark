import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import os

import json

from skimage.metrics import structural_similarity as ssim

def compute_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10 * np.log10(data_range ** 2 / mse)

def compute_ssim(gt, recon):
    """Structural similarity index."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)

def compute_cc(gt, recon):
    """Pearson correlation coefficient."""
    g = gt.ravel()
    r = recon.ravel()
    g_m = g - g.mean()
    r_m = r - r.mean()
    num = np.sum(g_m * r_m)
    den = np.sqrt(np.sum(g_m ** 2) * np.sum(r_m ** 2))
    if den < 1e-15:
        return 0.0
    return float(num / den)

def compute_dynamic_range(image, source_mask):
    """Ratio of peak signal to rms in background region."""
    bg = image[~source_mask]
    rms = np.sqrt(np.mean(bg ** 2)) if len(bg) > 0 else 1e-15
    if rms < 1e-15:
        rms = 1e-15
    return float(image.max() / rms)

def make_figure(sky_gt, dirty, recon, u, v, save_path):
    """Create 5-panel figure: GT, dirty, recon, error, uv-coverage."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    vmin_log = max(sky_gt[sky_gt > 0].min() * 0.1, 1e-4) if np.any(sky_gt > 0) else 1e-4
    vmax = sky_gt.max()

    ax = axes[0, 0]
    im = ax.imshow(sky_gt, origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=vmin_log, vmax=vmax))
    ax.set_title('(a) Ground Truth Sky', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[0, 1]
    im = ax.imshow(dirty, origin='lower', cmap='inferno')
    ax.set_title('(b) Dirty Image', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[0, 2]
    im = ax.imshow(recon, origin='lower', cmap='inferno',
                   vmin=0, vmax=vmax)
    ax.set_title('(c) L1+TSV Reconstruction', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[1, 0]
    error = np.abs(sky_gt - recon)
    im = ax.imshow(error, origin='lower', cmap='hot')
    ax.set_title('(d) Error |GT - Recon|', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='|Error|')

    ax = axes[1, 1]
    ax.scatter(u, v, s=0.3, alpha=0.3, color='cyan', edgecolors='none')
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlabel('u (pixels)', fontsize=11)
    ax.set_ylabel('v (pixels)', fontsize=11)
    ax.set_title('(e) (u,v) Coverage', fontsize=13, fontweight='bold')

    ax = axes[1, 2]
    cy, cx = 64, 64
    ax.plot(sky_gt[cy, :], 'b-', linewidth=1.5, label='GT row')
    ax.plot(recon[cy, :], 'r--', linewidth=1.5, label='Recon row')
    ax.plot(sky_gt[:, cx], 'b:', linewidth=1.5, label='GT col')
    ax.plot(recon[:, cx], 'r:', linewidth=1.5, label='Recon col')
    ax.set_xlabel('Pixel', fontsize=11)
    ax.set_ylabel('Flux', fontsize=11)
    ax.set_title('(f) Cross-section (row/col through center)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('Task 183: Radio Interferometric Imaging (L1+TSV Sparse Reconstruction)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {save_path}")

def evaluate_results(sky_gt, recon, dirty, u_unique, v_unique, ui, 
                     lambda_l1, lambda_tsv, max_iter, results_dir='results'):
    """
    Compute metrics, save results, and create visualizations.
    
    Parameters:
        sky_gt: ground truth sky image
        recon: reconstructed image
        dirty: dirty image
        u_unique, v_unique: UV coordinates
        ui: grid indices (for counting unique points)
        lambda_l1, lambda_tsv: regularization parameters
        max_iter: number of iterations used
        results_dir: directory to save results
    
    Returns:
        metrics: dict with evaluation metrics
    """
    print("Step 6: Computing metrics ...")
    
    os.makedirs(results_dir, exist_ok=True)
    nx, ny = sky_gt.shape[1], sky_gt.shape[0]
    
    # Compute metrics
    psnr_val = compute_psnr(sky_gt, recon)
    ssim_val = compute_ssim(sky_gt, recon)
    cc_val = compute_cc(sky_gt, recon)

    # Source mask for dynamic range (pixels > 5% of max)
    source_mask = sky_gt > 0.05 * sky_gt.max()
    dr_val = compute_dynamic_range(recon, source_mask)

    # Dirty image metrics for comparison
    dirty_norm = np.maximum(dirty, 0)
    dirty_norm = dirty_norm / dirty_norm.max() * sky_gt.max() if dirty_norm.max() > 0 else dirty_norm
    psnr_dirty = compute_psnr(sky_gt, dirty_norm)

    print(f"  PSNR (reconstruction): {psnr_val:.2f} dB")
    print(f"  PSNR (dirty image):    {psnr_dirty:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  CC:   {cc_val:.4f}")
    print(f"  Dynamic Range: {dr_val:.1f}")

    metrics = {
        'task_id': 183,
        'task_name': 'priism_radio',
        'method': 'ISTA with L1+TSV regularization',
        'PSNR_dB': round(psnr_val, 2),
        'SSIM': round(ssim_val, 4),
        'CC': round(cc_val, 4),
        'dynamic_range': round(dr_val, 1),
        'PSNR_dirty_dB': round(psnr_dirty, 2),
        'n_uv_points': int(len(ui)),
        'image_size': [nx, ny],
        'lambda_l1': lambda_l1,
        'lambda_tsv': lambda_tsv,
        'max_iter': max_iter,
    }

    # Save metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Create visualization
    print("Step 7: Creating visualization ...")
    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    make_figure(sky_gt, dirty, recon, u_unique, v_unique, fig_path)

    # Save arrays
    print("Step 8: Saving arrays ...")
    np.save(os.path.join(results_dir, 'ground_truth.npy'), sky_gt)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon)
    np.save(os.path.join(results_dir, 'dirty_image.npy'), dirty)
    np.save(os.path.join(results_dir, 'uv_coords.npy'), np.stack([u_unique, v_unique]))

    print("=" * 60)
    print("DONE. All outputs saved to results/")
    print(f"  PSNR = {psnr_val:.2f} dB  (target > 20 dB)")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  CC   = {cc_val:.4f}")

    # Validation
    assert psnr_val > 20.0, f"PSNR {psnr_val:.2f} dB < 20 dB target!"
    print("✓ PSNR > 20 dB — PASS")
    
    return metrics
