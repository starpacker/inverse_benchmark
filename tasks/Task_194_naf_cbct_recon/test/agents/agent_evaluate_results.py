import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_results(gt, recon, recon_full, sino_full, sino_sparse, 
                     angles_full, angles_sparse, out_dir):
    """
    Compute metrics (PSNR, SSIM, RMSE) and save visualization.
    
    Args:
        gt: ground truth 3D volume
        recon: sparse-view reconstructed 3D volume
        recon_full: full-view reconstructed 3D volume
        sino_full: full sinograms
        sino_sparse: sparse sinograms
        angles_full: full projection angles
        angles_sparse: sparse projection angles
        out_dir: output directory for saving results
        
    Returns:
        metrics_sparse: dict with PSNR, SSIM, RMSE for sparse reconstruction
        metrics_full: dict with PSNR, SSIM, RMSE for full reconstruction
    """
    def compute_metrics(gt_vol, recon_vol):
        """Compute PSNR, SSIM, RMSE on [0,1]-normalized data."""
        gt_f = gt_vol.astype(np.float64)
        recon_f = recon_vol.astype(np.float64)

        vmin, vmax = gt_f.min(), gt_f.max()
        if vmax - vmin > 1e-10:
            gt_n = (gt_f - vmin) / (vmax - vmin)
            recon_n = np.clip((recon_f - vmin) / (vmax - vmin), 0, 1)
        else:
            gt_n, recon_n = gt_f, recon_f

        rmse_val = np.sqrt(np.mean((gt_n - recon_n) ** 2))
        psnr_list, ssim_list = [], []
        for iz in range(gt_n.shape[0]):
            gs, rs = gt_n[iz], recon_n[iz]
            if gs.max() - gs.min() < 1e-8:
                continue
            psnr_list.append(psnr(gs, rs, data_range=1.0))
            ssim_list.append(ssim(gs, rs, data_range=1.0))

        return {
            'psnr': round(np.mean(psnr_list) if psnr_list else 0.0, 4),
            'ssim': round(np.mean(ssim_list) if ssim_list else 0.0, 4),
            'rmse': round(float(rmse_val), 6)
        }

    # Compute metrics
    metrics_full = compute_metrics(gt, recon_full)
    metrics_sparse = compute_metrics(gt, recon)

    # Visualization
    D, H, W = gt.shape
    md, mh, mw = D // 2, H // 2, W // 2

    vmin, vmax = gt.min(), gt.max()
    norm = lambda x: np.clip((x - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else x

    gd, rfd, rsd = norm(gt), norm(recon_full), norm(recon)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        'Sparse-View Cone-Beam CT Reconstruction\n'
        f'Sparse ({len(angles_sparse)} views): PSNR={metrics_sparse["psnr"]:.2f} dB, SSIM={metrics_sparse["ssim"]:.4f}  |  '
        f'Full ({len(angles_full)} views): PSNR={metrics_full["psnr"]:.2f} dB, SSIM={metrics_full["ssim"]:.4f}',
        fontsize=14, fontweight='bold')

    views = [('Axial', lambda x: x[md], 'equal'),
             ('Coronal', lambda x: x[:, mh, :], 'auto'),
             ('Sagittal', lambda x: x[:, :, mw], 'auto')]

    for row, (name, sl, asp) in enumerate(views):
        axes[row, 0].imshow(sl(gd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 0].set_title(f'GT ({name})'); axes[row, 0].axis('off')

        axes[row, 1].imshow(sl(rfd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 1].set_title(f'Full Recon ({name})'); axes[row, 1].axis('off')

        axes[row, 2].imshow(sl(rsd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 2].set_title(f'Sparse Recon ({name})'); axes[row, 2].axis('off')

        err = np.abs(sl(gd) - sl(rsd))
        im = axes[row, 3].imshow(err, cmap='hot', vmin=0, vmax=0.3, aspect=asp)
        axes[row, 3].set_title(f'Error ({name})'); axes[row, 3].axis('off')
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    # Sinograms
    axes[3, 0].imshow(sino_full[md], cmap='gray', aspect='auto')
    axes[3, 0].set_title(f'Full Sinogram ({len(angles_full)} ang)'); axes[3, 0].set_xlabel('Angle')

    axes[3, 1].imshow(sino_sparse[md], cmap='gray', aspect='auto')
    axes[3, 1].set_title(f'Sparse Sinogram ({len(angles_sparse)} ang)'); axes[3, 1].set_xlabel('Angle')

    axes[3, 2].plot(gd[md, mh], 'k-', lw=2, label='GT')
    axes[3, 2].plot(rfd[md, mh], 'b--', lw=1.5, label=f'Full ({len(angles_full)})')
    axes[3, 2].plot(rsd[md, mh], 'r-.', lw=1.5, label=f'Sparse ({len(angles_sparse)})')
    axes[3, 2].legend(fontsize=8); axes[3, 2].set_title('Line Profile'); axes[3, 2].grid(alpha=0.3)

    axes[3, 3].axis('off')
    txt = (f"Volume: {D}x{H}x{W}\n\n"
           f"Full ({len(angles_full)} ang):\n  PSNR={metrics_full['psnr']:.2f}dB\n  SSIM={metrics_full['ssim']:.4f}\n  RMSE={metrics_full['rmse']:.6f}\n\n"
           f"Sparse ({len(angles_sparse)} ang):\n  PSNR={metrics_sparse['psnr']:.2f}dB\n  SSIM={metrics_sparse['ssim']:.4f}\n  RMSE={metrics_sparse['rmse']:.6f}\n\n"
           f"Method: FBP + SART")
    axes[3, 3].text(0.1, 0.95, txt, transform=axes[3, 3].transAxes, fontsize=11, va='top',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    vis_path = os.path.join(out_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {vis_path}")

    # Save metrics JSON
    out = {
        'task': 'naf_cbct_recon', 'task_id': 194, 'method': 'FBP_SART',
        'volume_size': D, 'num_angles_full': len(angles_full), 'num_angles_sparse': len(angles_sparse),
        'psnr': metrics_sparse['psnr'], 'ssim': metrics_sparse['ssim'], 'rmse': metrics_sparse['rmse'],
        'full_view_psnr': metrics_full['psnr'], 'full_view_ssim': metrics_full['ssim'], 'full_view_rmse': metrics_full['rmse'],
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Save arrays
    np.save(os.path.join(out_dir, 'ground_truth.npy'), gt.astype(np.float32))
    np.save(os.path.join(out_dir, 'reconstruction.npy'), recon.astype(np.float32))
    np.save(os.path.join(out_dir, 'reconstruction_full.npy'), recon_full.astype(np.float32))

    return metrics_sparse, metrics_full
