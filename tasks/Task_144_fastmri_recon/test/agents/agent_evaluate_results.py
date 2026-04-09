import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_results(gt, zero_filled, cs_recon, metrics_zf, metrics_cs, save_path):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    gt_disp = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    zf_disp = (zero_filled - zero_filled.min()) / (zero_filled.max() - zero_filled.min() + 1e-12)
    cs_disp = (cs_recon - cs_recon.min()) / (cs_recon.max() - cs_recon.min() + 1e-12)
    error_map = np.abs(gt_disp - cs_disp)

    axes[0].imshow(gt_disp, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(zf_disp, cmap='gray')
    axes[1].set_title(f'Zero-filled\nPSNR={metrics_zf["PSNR"]:.2f} SSIM={metrics_zf["SSIM"]:.3f}',
                      fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(cs_disp, cmap='gray')
    axes[2].set_title(f'CS-TV Recon (ISTA)\nPSNR={metrics_cs["PSNR"]:.2f} SSIM={metrics_cs["SSIM"]:.3f}',
                      fontsize=11)
    axes[2].axis('off')

    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.15)
    axes[3].set_title(f'Error Map (CS)\nRMSE={metrics_cs["RMSE"]:.4f}', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('FastMRI Reconstruction: Accelerated MRI from Undersampled k-Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")

def evaluate_results(gt_image, recon_cs, recon_zf, params, save_dir=None):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE for both CS and zero-filled reconstructions.
    Saves metrics, visualization, and numpy arrays.
    
    Args:
        gt_image: ground truth image (numpy array)
        recon_cs: CS-TV reconstruction (numpy array)
        recon_zf: zero-filled reconstruction (numpy array)
        params: dictionary with parameters from load_and_preprocess_data
        save_dir: directory to save results (optional)
    
    Returns:
        all_metrics: dictionary with all evaluation metrics
    """
    print("\n[evaluate_results] Computing metrics...")
    
    # Normalize for comparison
    gt_norm = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min() + 1e-12)
    cs_norm = (recon_cs - recon_cs.min()) / (recon_cs.max() - recon_cs.min() + 1e-12)
    zf_norm = (recon_zf - recon_zf.min()) / (recon_zf.max() - recon_zf.min() + 1e-12)
    
    # CS metrics
    psnr_cs = psnr(gt_norm, cs_norm, data_range=1.0)
    ssim_cs = ssim(gt_norm, cs_norm, data_range=1.0)
    rmse_cs = np.sqrt(np.mean((gt_norm - cs_norm) ** 2))
    
    metrics_cs = {
        'PSNR': float(psnr_cs),
        'SSIM': float(ssim_cs),
        'RMSE': float(rmse_cs)
    }
    
    # Zero-filled metrics
    psnr_zf = psnr(gt_norm, zf_norm, data_range=1.0)
    ssim_zf = ssim(gt_norm, zf_norm, data_range=1.0)
    rmse_zf = np.sqrt(np.mean((gt_norm - zf_norm) ** 2))
    
    metrics_zf = {
        'PSNR': float(psnr_zf),
        'SSIM': float(ssim_zf),
        'RMSE': float(rmse_zf)
    }
    
    print(f"  Zero-filled: PSNR={metrics_zf['PSNR']:.2f} dB, "
          f"SSIM={metrics_zf['SSIM']:.4f}, RMSE={metrics_zf['RMSE']:.4f}")
    print(f"  CS-TV Recon: PSNR={metrics_cs['PSNR']:.2f} dB, "
          f"SSIM={metrics_cs['SSIM']:.4f}, RMSE={metrics_cs['RMSE']:.4f}")
    
    # Compile all metrics
    all_metrics = {
        'task': 'fastmri_recon',
        'method': 'ISTA-TV Compressed Sensing MRI Reconstruction',
        'acceleration': params.get('acceleration', 4),
        'image_size': params.get('image_size', 128),
        'PSNR': metrics_cs['PSNR'],
        'SSIM': metrics_cs['SSIM'],
        'RMSE': metrics_cs['RMSE'],
        'zero_filled_PSNR': metrics_zf['PSNR'],
        'zero_filled_SSIM': metrics_zf['SSIM'],
        'zero_filled_RMSE': metrics_zf['RMSE'],
        'sampling_ratio': params.get('sampling_ratio', 0.25),
    }
    
    # Save results if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics JSON
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_path}")
        
        # Save visualization
        vis_path = os.path.join(save_dir, 'reconstruction_result.png')
        visualize_results(gt_image, recon_zf, recon_cs, metrics_zf, metrics_cs, vis_path)
        
        # Save numpy arrays
        np.save(os.path.join(save_dir, 'ground_truth.npy'), gt_image)
        np.save(os.path.join(save_dir, 'reconstruction.npy'), recon_cs)
        print(f"  Saved ground_truth.npy and reconstruction.npy")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CS-TV PSNR:  {metrics_cs['PSNR']:.2f} dB")
    print(f"  CS-TV SSIM:  {metrics_cs['SSIM']:.4f}")
    print(f"  CS-TV RMSE:  {metrics_cs['RMSE']:.4f}")
    print(f"  ZF PSNR:     {metrics_zf['PSNR']:.2f} dB")
    print(f"  ZF SSIM:     {metrics_zf['SSIM']:.4f}")
    
    # Quality check
    if metrics_cs['PSNR'] > 15 and metrics_cs['SSIM'] > 0.5:
        print("\n  ✓ Metrics PASS quality thresholds (PSNR>15, SSIM>0.5)")
    else:
        print("\n  ✗ Metrics BELOW quality thresholds - may need tuning")
    
    print("=" * 60)
    
    return all_metrics
