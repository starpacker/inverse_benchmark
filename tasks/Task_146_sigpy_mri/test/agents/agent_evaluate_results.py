import matplotlib

matplotlib.use('Agg')

import os

import sys

import json

import numpy as np

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, 'repo')

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_results(phantom, recon, zero_filled, config, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE metrics and generates visualization.
    
    Args:
        phantom: ground truth image (ny, nx)
        recon: L1-Wavelet reconstructed image (ny, nx)
        zero_filled: zero-filled reconstruction (ny, nx)
        config: dictionary containing reconstruction configuration
        results_dir: directory to save results
        
    Returns:
        dict containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Helper function to compute metrics
    def compute_metrics(gt, rec):
        gt_abs = np.abs(gt).astype(np.float64)
        recon_abs = np.abs(rec).astype(np.float64)
        
        gt_max = gt_abs.max()
        if gt_max > 0:
            gt_norm = gt_abs / gt_max
            recon_norm = recon_abs / gt_max
        else:
            gt_norm = gt_abs
            recon_norm = recon_abs
        
        recon_norm = np.clip(recon_norm, 0, recon_norm.max())
        
        psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
        ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)
        rmse_val = np.sqrt(np.mean((gt_norm - recon_norm) ** 2))
        
        return psnr_val, ssim_val, rmse_val
    
    # Compute metrics for both reconstructions
    psnr_recon, ssim_recon, rmse_recon = compute_metrics(phantom, recon)
    psnr_zf, ssim_zf, rmse_zf = compute_metrics(phantom, zero_filled)
    
    print(f"  Zero-filled — PSNR: {psnr_zf:.2f} dB, SSIM: {ssim_zf:.4f}, RMSE: {rmse_zf:.4f}")
    print(f"  L1-Wavelet — PSNR: {psnr_recon:.2f} dB, SSIM: {ssim_recon:.4f}, RMSE: {rmse_recon:.4f}")
    
    # Quality check
    print("\n" + "=" * 60)
    psnr_ok = psnr_recon > 15
    ssim_ok = ssim_recon > 0.5
    print(f"PSNR > 15: {'PASS' if psnr_ok else 'FAIL'} ({psnr_recon:.2f} dB)")
    print(f"SSIM > 0.5: {'PASS' if ssim_ok else 'FAIL'} ({ssim_recon:.4f})")
    print(f"Improvement over zero-filled: +{psnr_recon - psnr_zf:.2f} dB PSNR")
    
    # Create visualization
    gt_abs = np.abs(phantom).astype(np.float64)
    zf_abs = np.abs(zero_filled).astype(np.float64)
    recon_abs = np.abs(recon).astype(np.float64)
    
    gt_max = gt_abs.max()
    gt_norm = gt_abs / gt_max if gt_max > 0 else gt_abs
    zf_norm = zf_abs / gt_max if gt_max > 0 else zf_abs
    recon_norm = recon_abs / gt_max if gt_max > 0 else recon_abs
    
    error_map = np.abs(gt_norm - recon_norm)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(gt_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(zf_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Zero-filled\nPSNR={psnr_zf:.2f}, SSIM={ssim_zf:.4f}', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'L1-Wavelet Recon\nPSNR={psnr_recon:.2f}, SSIM={ssim_recon:.4f}', fontsize=12)
    axes[2].axis('off')
    
    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.1)
    axes[3].set_title('Error Map (|GT - Recon|)', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle('SigPy MRI Reconstruction: L1-Wavelet Compressed Sensing\n'
                 f'({config.get("num_coils", 8)}-Coil Parallel Imaging, '
                 f'{config.get("accel_factor", 4)}× Poisson Variable-Density Acceleration)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {vis_path}")
    
    # Save metrics JSON
    metrics = {
        'task': 'sigpy_mri',
        'method': 'L1-Wavelet Compressed Sensing (SigPy)',
        'library': 'sigpy',
        'forward_operator': f'2D FFT with Poisson variable-density undersampling '
                           f'({config.get("accel_factor", 4)}x acceleration, '
                           f'{config.get("num_coils", 8)}-coil parallel imaging)',
        'num_coils': config.get('num_coils', 8),
        'image_shape': list(config.get('img_shape', (128, 128))),
        'acceleration_factor': config.get('accel_factor', 4),
        'sampling_pattern': 'Poisson variable-density',
        'regularization_lambda': config.get('lamda', 0.001),
        'max_iterations': config.get('max_iter', 200),
        'wavelet': config.get('wavelet', 'db4'),
        'psnr': round(float(psnr_recon), 2),
        'ssim': round(float(ssim_recon), 4),
        'rmse': round(float(rmse_recon), 4),
        'zero_filled_psnr': round(float(psnr_zf), 2),
        'zero_filled_ssim': round(float(ssim_zf), 4),
        'zero_filled_rmse': round(float(rmse_zf), 4),
        'psnr_improvement': round(float(psnr_recon - psnr_zf), 2),
    }
    
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")
    
    # Save numpy arrays
    gt_path = os.path.join(results_dir, 'ground_truth.npy')
    recon_path = os.path.join(results_dir, 'reconstruction.npy')
    np.save(gt_path, np.abs(phantom))
    np.save(recon_path, np.abs(recon))
    print(f"  Ground truth saved to {gt_path}")
    print(f"  Reconstruction saved to {recon_path}")
    
    return metrics
