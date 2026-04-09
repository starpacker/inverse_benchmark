import sys

import os

import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

from skimage.metrics import structural_similarity

sys.path.insert(0, '/data/yjh/odtbrain_sandbox/repo')

RESULTS_DIR = '/data/yjh/odtbrain_sandbox/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(phantom_ri, ri_recon_aligned, sino_noisy, params, results_dir):
    """
    Compute metrics, create visualizations, and save results.
    
    Calculates PSNR, SSIM, RMSE for both 2D central slices and 3D volumes.
    Creates visualization plots and saves all results to disk.
    
    Parameters:
        phantom_ri: 3D numpy array - ground truth refractive index
        ri_recon_aligned: 3D numpy array - reconstructed refractive index
        sino_noisy: 3D complex numpy array - noisy sinogram (for visualization)
        params: dict - parameters used in reconstruction
        results_dir: str - path to output directory
    
    Returns:
        dict containing all computed metrics
    """
    N = params['N']
    nm = params['nm']
    n_sphere = params['n_sphere']
    num_angles = params['num_angles']
    noise_level = params['noise_level']
    
    # Compute PSNR
    def compute_psnr(gt, recon):
        mse = np.mean((gt - recon) ** 2)
        if mse == 0:
            return float('inf')
        data_range = gt.max() - gt.min()
        return 10 * np.log10(data_range ** 2 / mse)
    
    # Compute SSIM for 2D images
    def compute_ssim(gt, recon, data_range=None):
        if data_range is None:
            data_range = gt.max() - gt.min()
        return structural_similarity(gt, recon, data_range=data_range)
    
    # Compute RMSE
    def compute_rmse(gt, recon):
        return np.sqrt(np.mean((gt - recon) ** 2))
    
    # Use central slices for 2D metrics
    center = N // 2
    gt_slice = phantom_ri[center, :, :]
    recon_slice = ri_recon_aligned[center, :, :]
    
    data_range = phantom_ri.max() - phantom_ri.min()
    
    # 2D metrics on central slice
    psnr_2d = compute_psnr(gt_slice, recon_slice)
    ssim_2d = compute_ssim(gt_slice, recon_slice, data_range=data_range)
    rmse_2d = compute_rmse(gt_slice, recon_slice)
    
    # 3D metrics (volumetric)
    psnr_3d = compute_psnr(phantom_ri, ri_recon_aligned)
    ssim_slices = []
    for i in range(phantom_ri.shape[0]):
        s = compute_ssim(phantom_ri[i], ri_recon_aligned[i], data_range=data_range)
        ssim_slices.append(s)
    ssim_3d = float(np.mean(ssim_slices))
    rmse_3d = compute_rmse(phantom_ri, ri_recon_aligned)
    
    print(f"  Central slice - PSNR: {psnr_2d:.2f} dB, SSIM: {ssim_2d:.4f}, RMSE: {rmse_2d:.6f}")
    print(f"  3D volume     - PSNR: {psnr_3d:.2f} dB, SSIM: {ssim_3d:.4f}, RMSE: {rmse_3d:.6f}")
    
    metrics = {
        'PSNR': round(float(psnr_2d), 2),
        'SSIM': round(float(ssim_2d), 4),
        'RMSE': round(float(rmse_2d), 6),
        'PSNR_3D': round(float(psnr_3d), 2),
        'SSIM_3D': round(float(ssim_3d), 4),
        'RMSE_3D': round(float(rmse_3d), 6),
        'num_projections': num_angles,
        'grid_size': N,
        'noise_level': noise_level,
        'medium_index': nm,
        'sphere_index': n_sphere,
        'method': 'Rytov backpropagation (ODTbrain)'
    }
    
    # Save metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Ground truth, Sinogram, Reconstruction
    im0 = axes[0, 0].imshow(gt_slice, vmin=nm-0.001, vmax=n_sphere+0.001,
                              cmap='hot', interpolation='none')
    axes[0, 0].set_title('Ground Truth (central xz slice)', fontsize=11)
    axes[0, 0].set_xlabel('x [px]')
    axes[0, 0].set_ylabel('z [px]')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, format='%.4f')
    
    # Sinogram (phase of one projection)
    sino_phase = np.angle(sino_noisy)
    im1 = axes[0, 1].imshow(sino_phase[:, center, :],
                              aspect=sino_noisy.shape[2]/sino_noisy.shape[0],
                              cmap='coolwarm', interpolation='none')
    axes[0, 1].set_title('Phase Sinogram (y=center)', fontsize=11)
    axes[0, 1].set_xlabel('detector x [px]')
    axes[0, 1].set_ylabel('angle index')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Reconstruction central slice
    im2 = axes[0, 2].imshow(recon_slice, vmin=nm-0.001, vmax=n_sphere+0.001,
                              cmap='hot', interpolation='none')
    axes[0, 2].set_title(f'Reconstruction (PSNR={psnr_2d:.1f}dB, SSIM={ssim_2d:.3f})',
                          fontsize=11)
    axes[0, 2].set_xlabel('x [px]')
    axes[0, 2].set_ylabel('z [px]')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, format='%.4f')
    
    # Row 2: Error map, line profile, other slices
    error = np.abs(gt_slice - recon_slice)
    im3 = axes[1, 0].imshow(error, cmap='hot', interpolation='none')
    axes[1, 0].set_title('Absolute Error Map', fontsize=11)
    axes[1, 0].set_xlabel('x [px]')
    axes[1, 0].set_ylabel('z [px]')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, format='%.5f')
    
    # Line profile through center
    center_line_gt = phantom_ri[center, center, :]
    center_line_recon = ri_recon_aligned[center, center, :]
    axes[1, 1].plot(center_line_gt, 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(center_line_recon, 'r--', linewidth=2, label='Reconstruction')
    axes[1, 1].set_title('Line Profile (through center)', fontsize=11)
    axes[1, 1].set_xlabel('x [px]')
    axes[1, 1].set_ylabel('Refractive Index')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([nm - 0.002, n_sphere + 0.002])
    
    # YZ reconstruction slice
    im5 = axes[1, 2].imshow(ri_recon_aligned[:, :, center].T,
                              vmin=nm-0.001, vmax=n_sphere+0.001,
                              cmap='hot', interpolation='none')
    axes[1, 2].set_title('Reconstruction (yz slice at x=center)', fontsize=11)
    axes[1, 2].set_xlabel('z [px]')
    axes[1, 2].set_ylabel('y [px]')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, format='%.4f')
    
    plt.suptitle('ODTbrain: 3D RI Reconstruction via Rytov Backpropagation\n'
                 f'N={N}, {num_angles} projections, noise={noise_level}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {fig_path}")
    
    # Save numpy arrays
    gt_path = os.path.join(results_dir, 'ground_truth.npy')
    recon_path = os.path.join(results_dir, 'reconstruction.npy')
    np.save(gt_path, phantom_ri)
    np.save(recon_path, ri_recon_aligned)
    print(f"  Ground truth saved to {gt_path}")
    print(f"  Reconstruction saved to {recon_path}")
    
    return metrics
