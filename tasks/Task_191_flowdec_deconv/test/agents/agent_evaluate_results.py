import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

def evaluate_results(gt_volume, observed, deconvolved, save_dir='results'):
    """
    Evaluate reconstruction quality and visualize results.
    
    Parameters
    ----------
    gt_volume : np.ndarray
        Ground truth 3D volume.
    observed : np.ndarray
        Blurred + noisy observation.
    deconvolved : np.ndarray
        Reconstructed 3D volume.
    save_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR and SSIM values.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    mid_z = gt_volume.shape[0] // 2

    gt_slice = gt_volume[mid_z]
    recon_slice = deconvolved[mid_z]
    obs_slice = observed[mid_z]

    # Normalize both to [0, 1] using GT range for fair comparison
    vmin, vmax = gt_slice.min(), gt_slice.max()
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0

    gt_norm = (gt_slice - vmin) / (vmax - vmin)
    recon_norm = np.clip((recon_slice - vmin) / (vmax - vmin), 0, 1)
    obs_norm = np.clip((obs_slice - vmin) / (vmax - vmin), 0, 1)

    psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)
    
    # Baseline metrics for blurred input
    psnr_baseline = psnr(gt_norm, obs_norm, data_range=1.0)
    ssim_baseline = ssim(gt_norm, obs_norm, data_range=1.0)

    metrics = {
        'psnr': float(round(psnr_val, 4)), 
        'ssim': float(round(ssim_val, 4)),
        'baseline_psnr': float(round(psnr_baseline, 4)),
        'baseline_ssim': float(round(ssim_baseline, 4))
    }
    
    # Visualize central z-slice
    gt_s = gt_volume[mid_z]
    obs_s = observed[mid_z]
    dec_s = deconvolved[mid_z]
    err_s = np.abs(gt_s - dec_s)

    vmin_vis, vmax_vis = gt_s.min(), gt_s.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(gt_s, cmap='hot', vmin=vmin_vis, vmax=vmax_vis)
    axes[0].set_title('Ground Truth (z-center)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(obs_s, cmap='hot', vmin=vmin_vis, vmax=vmax_vis)
    axes[1].set_title('Blurred + Noisy Input', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(dec_s, cmap='hot', vmin=vmin_vis, vmax=vmax_vis)
    axes[2].set_title(f'RL Deconvolved\nPSNR={metrics["psnr"]:.2f} dB, SSIM={metrics["ssim"]:.4f}',
                       fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_s, cmap='viridis')
    axes[3].set_title('|GT − Deconvolved| Error', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('Task 191: 3D Richardson-Lucy Deconvolution (Fluorescence Microscopy)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

    # Save metrics to JSON
    metrics_out = {
        'task': 'flowdec_deconv',
        'task_number': 191,
        'method': 'Richardson-Lucy deconvolution (scikit-image)',
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'baseline_psnr': metrics['baseline_psnr'],
        'baseline_ssim': metrics['baseline_ssim']
    }
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save arrays
    np.save(os.path.join(save_dir, 'ground_truth.npy'), gt_volume)
    np.save(os.path.join(save_dir, 'reconstruction.npy'), deconvolved)
    print(f"Arrays saved to {save_dir}/")

    return metrics
