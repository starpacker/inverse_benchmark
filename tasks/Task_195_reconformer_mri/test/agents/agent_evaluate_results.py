import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_metric

from skimage.metrics import peak_signal_noise_ratio as psnr_metric

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(data, inversion_result):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR, SSIM, RMSE metrics for:
      - Zero-filled baseline
      - Raw reconstruction
      - Intensity-corrected reconstruction
    
    Saves:
      - metrics.json
      - ground_truth.npy, reconstruction.npy, zero_filled.npy, mask.npy
      - reconstruction_result.png visualization
    
    Args:
        data: Dictionary from load_and_preprocess_data
        inversion_result: Dictionary from run_inversion
    
    Returns:
        dict: All computed metrics
    """
    gt_image = data['gt_image']
    zero_filled = data['zero_filled']
    mask = data['mask']
    
    recon_raw = inversion_result['recon_raw']
    final_recon = inversion_result['final_recon']
    
    def compute_metrics(gt, recon):
        data_range = gt.max() - gt.min() + 1e-12
        gt_norm = (gt - gt.min()) / data_range
        recon_norm = np.clip((recon - gt.min()) / data_range, 0, 1)
        
        p = float(psnr_metric(gt_norm, recon_norm, data_range=1.0))
        s = float(ssim_metric(gt_norm, recon_norm, data_range=1.0))
        r = float(np.sqrt(np.mean((gt_norm - recon_norm)**2)))
        
        return {'psnr': round(p, 4), 'ssim': round(s, 4), 'rmse': round(r, 6)}
    
    metrics_zf = compute_metrics(gt_image, zero_filled)
    metrics_raw = compute_metrics(gt_image, recon_raw)
    metrics_corrected = compute_metrics(gt_image, final_recon)
    
    print(f"\n  Zero-filled baseline: PSNR={metrics_zf['psnr']:.2f} dB, "
          f"SSIM={metrics_zf['ssim']:.4f}")
    print(f"  Raw reconstruction: PSNR={metrics_raw['psnr']:.2f} dB, "
          f"SSIM={metrics_raw['ssim']:.4f}")
    print(f"  Corrected: PSNR={metrics_corrected['psnr']:.2f} dB, "
          f"SSIM={metrics_corrected['ssim']:.4f}")
    
    all_metrics = {
        'task': 'reconformer_mri',
        'task_id': 195,
        'method': 'FISTA + TV (Compressed Sensing MRI)',
        'acceleration': 4,
        'image_size': data['N'],
        'sampling_rate': 0.25,
        'tv_lambda': 0.0003,
        'fista_iterations': 1300,
        'zero_filled': metrics_zf,
        'raw_reconstruction': metrics_raw,
        'corrected_reconstruction': metrics_corrected,
        'psnr': metrics_corrected['psnr'],
        'ssim': metrics_corrected['ssim'],
        'rmse': metrics_corrected['rmse'],
    }
    
    print("\n  Saving results...")
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")
    
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), gt_image)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), final_recon)
    np.save(os.path.join(RESULTS_DIR, 'zero_filled.npy'), zero_filled)
    np.save(os.path.join(RESULTS_DIR, 'mask.npy'), mask)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = gt_image.min(), gt_image.max()
    
    axes[0].imshow(gt_image, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('(a) Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(zero_filled, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'(b) Zero-filled\nPSNR={metrics_zf["psnr"]:.2f} dB, '
                      f'SSIM={metrics_zf["ssim"]:.4f}', fontsize=11)
    axes[1].axis('off')
    
    axes[2].imshow(final_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'(c) CS-TV Reconstruction\nPSNR={metrics_corrected["psnr"]:.2f} dB, '
                      f'SSIM={metrics_corrected["ssim"]:.4f}', fontsize=11)
    axes[2].axis('off')
    
    error = np.abs(gt_image - final_recon)
    im = axes[3].imshow(error, cmap='hot', vmin=0, vmax=max(error.max() * 0.5, 1e-6))
    axes[3].set_title('(d) Error Map |GT - Recon|', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    fig.suptitle('Accelerated MRI Reconstruction (4x Cartesian Undersampling)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    vis_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {vis_path}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Zero-filled:   PSNR={metrics_zf['psnr']:.2f} dB, SSIM={metrics_zf['ssim']:.4f}")
    print(f"  Raw CS-TV:     PSNR={metrics_raw['psnr']:.2f} dB, SSIM={metrics_raw['ssim']:.4f}")
    print(f"  Final (corr.): PSNR={metrics_corrected['psnr']:.2f} dB, SSIM={metrics_corrected['ssim']:.4f}")
    print(f"  RMSE:          {metrics_corrected['rmse']:.6f}")
    
    target_met = metrics_corrected['psnr'] > 25 and metrics_corrected['ssim'] > 0.85
    print(f"\n  Target (PSNR>25, SSIM>0.85): {'MET ✓' if target_met else 'NOT MET ✗'}")
    print("=" * 70)
    
    return all_metrics
