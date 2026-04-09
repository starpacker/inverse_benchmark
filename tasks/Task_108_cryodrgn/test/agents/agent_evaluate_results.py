import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim_metric

from skimage.metrics import peak_signal_noise_ratio as psnr_metric

def evaluate_results(ground_truth, reconstruction, projections, results_dir, script_dir,
                     n_projections, vol_size, noise_std):
    """
    Evaluate reconstruction quality and generate outputs.
    
    Metrics computed: PSNR, SSIM, RMSE
    
    Parameters:
    -----------
    ground_truth : np.ndarray
        The ground truth 3D volume.
    reconstruction : np.ndarray
        The reconstructed 3D volume.
    projections : np.ndarray
        The 2D projection images (for visualization).
    results_dir : str
        Directory to save results.
    script_dir : str
        Directory for saving normalized outputs.
    n_projections : int
        Number of projections used.
    vol_size : int
        Size of the volume.
    noise_std : float
        Noise standard deviation used.
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    # Normalize volumes for metric computation
    gt_n = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-10)
    rec_n = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-10)
    
    # Compute RMSE
    rmse = float(np.sqrt(np.mean((gt_n - rec_n) ** 2)))
    
    # Compute per-axis SSIM and PSNR
    N = ground_truth.shape[0]
    mid = N // 2
    slices = [
        (gt_n[mid, :, :], rec_n[mid, :, :]),
        (gt_n[:, mid, :], rec_n[:, mid, :]),
        (gt_n[:, :, mid], rec_n[:, :, mid])
    ]
    ss = [float(ssim_metric(g, r, data_range=1.0)) for g, r in slices]
    ps = [float(psnr_metric(g, r, data_range=1.0)) for g, r in slices]
    
    # Compute 3D PSNR
    psnr_3d = float(psnr_metric(gt_n, rec_n, data_range=1.0))
    
    metrics = {
        'PSNR_dB': round(psnr_3d, 4),
        'SSIM': round(float(np.mean(ss)), 4),
        'RMSE': round(rmse, 6),
        'PSNR_per_axis': [round(v, 4) for v in ps],
        'SSIM_per_axis': [round(v, 4) for v in ss],
        'n_projections': n_projections,
        'volume_size': vol_size,
        'noise_std': noise_std,
    }
    
    print(f"  PSNR: {metrics['PSNR_dB']:.2f} dB")
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    
    # Save metrics and data
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), reconstruction)
    
    # Save normalized outputs
    np.save(os.path.join(script_dir, 'gt_output.npy'), gt_n)
    np.save(os.path.join(script_dir, 'recon_output.npy'), rec_n)
    print(f"  Saved gt_output.npy: range [{gt_n.min():.4f}, {gt_n.max():.4f}]")
    print(f"  Saved recon_output.npy: range [{rec_n.min():.4f}, {rec_n.max():.4f}]")
    
    # Generate visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # Ground truth slices
    axes[0, 0].imshow(gt_n[mid, :, :], cmap='gray')
    axes[0, 0].set_title('GT: Axial')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gt_n[:, mid, :], cmap='gray')
    axes[0, 1].set_title('GT: Coronal')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_n[:, :, mid], cmap='gray')
    axes[0, 2].set_title('GT: Sagittal')
    axes[0, 2].axis('off')
    axes[0, 3].imshow(np.max(gt_n, axis=0), cmap='hot')
    axes[0, 3].set_title('GT: MIP')
    axes[0, 3].axis('off')
    
    # Sample projections
    for j in range(4):
        idx = j * (len(projections) // 4)
        axes[1, j].imshow(projections[idx], cmap='gray')
        axes[1, j].set_title(f'Proj #{idx}')
        axes[1, j].axis('off')
    
    # Reconstruction slices
    axes[2, 0].imshow(rec_n[mid, :, :], cmap='gray')
    axes[2, 0].set_title('Recon: Axial')
    axes[2, 0].axis('off')
    axes[2, 1].imshow(rec_n[:, mid, :], cmap='gray')
    axes[2, 1].set_title('Recon: Coronal')
    axes[2, 1].axis('off')
    axes[2, 2].imshow(rec_n[:, :, mid], cmap='gray')
    axes[2, 2].set_title('Recon: Sagittal')
    axes[2, 2].axis('off')
    axes[2, 3].imshow(np.max(rec_n, axis=0), cmap='hot')
    axes[2, 3].set_title('Recon: MIP')
    axes[2, 3].axis('off')
    
    # Error maps
    for j, (title, sl) in enumerate([('Axial', mid), ('Coronal', mid), ('Sagittal', mid)]):
        if j == 0:
            err = np.abs(gt_n[sl, :, :] - rec_n[sl, :, :])
        elif j == 1:
            err = np.abs(gt_n[:, sl, :] - rec_n[:, sl, :])
        else:
            err = np.abs(gt_n[:, :, sl] - rec_n[:, :, sl])
        im = axes[3, j].imshow(err, cmap='hot', vmin=0, vmax=0.5)
        axes[3, j].set_title(f'Error: {title}')
        axes[3, j].axis('off')
        plt.colorbar(im, ax=axes[3, j], fraction=0.046)
    
    # Metrics text
    axes[3, 3].axis('off')
    t = (f"PSNR: {metrics['PSNR_dB']:.2f} dB\n"
         f"SSIM: {metrics['SSIM']:.4f}\n"
         f"RMSE: {metrics['RMSE']:.6f}\n"
         f"N_proj: {metrics['n_projections']}\n"
         f"Vol: {metrics['volume_size']}^3\n"
         f"Noise: {metrics['noise_std']}")
    axes[3, 3].text(0.1, 0.5, t, transform=axes[3, 3].transAxes, fontsize=14,
                    va='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3, 3].set_title('Metrics')
    
    fig.suptitle('Task 118: CryoDRGN - Cryo-EM 3D Reconstruction\n'
                 'Forward: Fourier Slice + CTF + Noise | Inverse: CTF-Weighted DFI',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
