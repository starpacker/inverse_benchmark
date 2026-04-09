import matplotlib

matplotlib.use('Agg')

import json

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from skimage.metrics import peak_signal_noise_ratio as compute_psnr

from skimage.metrics import structural_similarity as compute_ssim

def visualize(gt, degraded, recon, metrics, save_path):
    """4-panel figure: GT | Degraded | Reconstruction | Error map."""
    error = np.abs(gt - recon)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ['Ground Truth', 'Degraded (Blur+Noise)',
              f'DPS Reconstruction\nPSNR={metrics["psnr_db"]:.2f} dB  '
              f'SSIM={metrics["ssim"]:.4f}',
              'Absolute Error']

    for ax, img, title in zip(axes, [gt, degraded, recon, error], titles):
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.colorbar(axes[3].images[0], ax=axes[3], fraction=0.046)

    plt.suptitle('Diffusion Posterior Sampling (DPS) — Image Deblurring',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Saved visualization to {save_path}")

def evaluate_results(gt_np: np.ndarray, degraded_np: np.ndarray,
                     recon_np: np.ndarray, config: dict,
                     results_dir: Path, elapsed_time: float):
    """
    Evaluate and save results.
    
    Args:
        gt_np: Ground truth image as numpy array
        degraded_np: Degraded image as numpy array
        recon_np: Reconstructed image as numpy array
        config: Configuration dictionary
        results_dir: Directory to save results
        elapsed_time: Total elapsed time
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n[6/6] Evaluating reconstruction ...")
    
    # Compute metrics for reconstruction
    gt_f = gt_np.astype(np.float64)
    recon_f = recon_np.astype(np.float64)
    psnr = compute_psnr(gt_f, recon_f, data_range=1.0)
    ssim = compute_ssim(gt_f, recon_f, data_range=1.0)
    rmse = np.sqrt(np.mean((gt_f - recon_f) ** 2))
    
    metrics = {'psnr_db': float(psnr), 'ssim': float(ssim), 'rmse': float(rmse)}
    
    print(f"  PSNR  = {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM  = {metrics['ssim']:.4f}")
    print(f"  RMSE  = {metrics['rmse']:.6f}")
    
    # Compute metrics for degraded image
    degraded_f = degraded_np.astype(np.float64)
    deg_psnr = compute_psnr(gt_f, degraded_f, data_range=1.0)
    deg_ssim = compute_ssim(gt_f, degraded_f, data_range=1.0)
    
    deg_metrics = {'psnr_db': float(deg_psnr), 'ssim': float(deg_ssim)}
    
    # Save metrics JSON
    metrics_out = {
        'psnr_db': metrics['psnr_db'],
        'ssim': metrics['ssim'],
        'rmse': metrics['rmse'],
        'degraded_psnr_db': deg_metrics['psnr_db'],
        'degraded_ssim': deg_metrics['ssim'],
        'method': 'Diffusion Posterior Sampling (DPS)',
        'inverse_problem': 'Gaussian deblurring',
        'image_size': config['img_size'],
        'diffusion_steps': config['num_timesteps'],
        'blur_sigma': config['blur_sigma'],
        'noise_std': config['noise_std'],
        'elapsed_seconds': elapsed_time,
    }
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")
    
    # Save numpy arrays
    np.save(results_dir / 'ground_truth.npy', gt_np)
    np.save(results_dir / 'reconstruction.npy', recon_np)
    np.save(results_dir / 'degraded.npy', degraded_np)
    print(f"  Saved .npy arrays to {results_dir}")
    
    # Visualization
    vis_path = results_dir / 'reconstruction_result.png'
    visualize(gt_np, degraded_np, recon_np, metrics, vis_path)
    
    print(f"\n{'='*60}")
    print(f" DPS Deblurring Complete")
    print(f" PSNR = {metrics['psnr_db']:.2f} dB   SSIM = {metrics['ssim']:.4f}")
    print(f" Elapsed: {elapsed_time:.1f}s")
    print(f"{'='*60}\n")
    
    return metrics
