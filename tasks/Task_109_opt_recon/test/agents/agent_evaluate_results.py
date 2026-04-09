import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_results(phantom, reconstruction, sinograms_noisy, theta, params, output_dir='results'):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE metrics for both the full 3D volume and middle slice.
    Saves metrics to JSON, arrays to NPY, and creates visualization.
    
    Args:
        phantom: 3D ground truth array
        reconstruction: 3D reconstructed array
        sinograms_noisy: 3D array of noisy sinograms
        theta: projection angles
        params: dictionary of parameters
        output_dir: directory to save results
    
    Returns:
        metrics: dictionary containing all computed metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    nz, ny, nx = phantom.shape
    mid_z = nz // 2
    
    # Compute 3D volume metrics
    data_range_3d = phantom.max() - phantom.min()
    if data_range_3d < 1e-10:
        data_range_3d = 1.0
    
    psnr_3d = peak_signal_noise_ratio(phantom, reconstruction, data_range=data_range_3d)
    ssim_3d = structural_similarity(phantom, reconstruction, data_range=data_range_3d)
    rmse_3d = np.sqrt(np.mean((phantom - reconstruction) ** 2))
    
    # Compute middle slice metrics
    gt_mid = phantom[mid_z]
    recon_mid = reconstruction[mid_z]
    data_range_mid = gt_mid.max() - gt_mid.min()
    if data_range_mid < 1e-10:
        data_range_mid = 1.0
    
    psnr_mid = peak_signal_noise_ratio(gt_mid, recon_mid, data_range=data_range_mid)
    ssim_mid = structural_similarity(gt_mid, recon_mid, data_range=data_range_mid)
    rmse_mid = np.sqrt(np.mean((gt_mid - recon_mid) ** 2))
    
    # Build metrics dictionary
    metrics = {
        'task': 'opt_recon',
        'task_number': 119,
        'method': 'Filtered Back-Projection (FBP)',
        'inverse_problem': 'Optical Projection Tomography (OPT) 3D Reconstruction',
        'phantom_shape': list(phantom.shape),
        'n_angles': params['n_angles'],
        'photon_count': params['photon_count'],
        'readout_noise_std': params['readout_std'],
        'metrics_3d': {
            'PSNR_dB': round(psnr_3d, 2),
            'SSIM': round(ssim_3d, 4),
            'RMSE': round(rmse_3d, 6)
        },
        'metrics_middle_slice': {
            'PSNR_dB': round(psnr_mid, 2),
            'SSIM': round(ssim_mid, 4),
            'RMSE': round(rmse_mid, 6)
        }
    }
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(output_dir, 'ground_truth.npy'), phantom)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), reconstruction)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Task 119: OPT Reconstruction (Filtered Back-Projection)',
                 fontsize=16, fontweight='bold')
    
    gt_slice = phantom[mid_z]
    sino_slice = sinograms_noisy[mid_z]
    recon_slice = reconstruction[mid_z]
    error_map = np.abs(gt_slice - recon_slice)
    
    vmin, vmax = 0, gt_slice.max()
    
    # GT slice
    im0 = axes[0, 0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Ground Truth (z={mid_z})', fontsize=13)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Sinogram
    im1 = axes[0, 1].imshow(sino_slice.T, cmap='gray', aspect='auto',
                             extent=[0, sino_slice.shape[0], 180, 0])
    axes[0, 1].set_title(f'Sinogram (noisy, z={mid_z})', fontsize=13)
    axes[0, 1].set_xlabel('Detector position')
    axes[0, 1].set_ylabel('Angle (degrees)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Reconstruction
    im2 = axes[0, 2].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'FBP Reconstruction (z={mid_z})', fontsize=13)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Error map
    im3 = axes[1, 0].imshow(error_map, cmap='hot')
    axes[1, 0].set_title('Absolute Error Map', fontsize=13)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Line profile comparison
    mid_y = ny // 2
    axes[1, 1].plot(gt_slice[mid_y, :], 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(recon_slice[mid_y, :], 'r--', linewidth=2, label='FBP Recon')
    axes[1, 1].set_title(f'Line Profile (y={mid_y})', fontsize=13)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3D overview: show multiple slices
    slice_indices = [nz // 4, nz // 2, 3 * nz // 4]
    for i, zi in enumerate(slice_indices):
        color = ['blue', 'green', 'red'][i]
        gt_zi = phantom[zi]
        recon_zi = reconstruction[zi]
        dr_zi = gt_zi.max() - gt_zi.min()
        if dr_zi < 1e-10:
            dr_zi = 1.0
        psnr_i = peak_signal_noise_ratio(gt_zi, recon_zi, data_range=dr_zi)
        axes[1, 2].plot(reconstruction[zi][ny // 2, :], color=color, alpha=0.7,
                        label=f'z={zi} (PSNR={psnr_i:.1f}dB)')
    axes[1, 2].set_title('Reconstruction Profiles at Different z-Slices', fontsize=13)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add metrics text
    fig.text(0.5, 0.01,
             f'3D Volume --- PSNR: {psnr_3d:.2f} dB | SSIM: {ssim_3d:.4f} | RMSE: {rmse_3d:.6f}',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
