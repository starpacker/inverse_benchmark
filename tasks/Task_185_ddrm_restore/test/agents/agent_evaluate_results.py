import matplotlib

matplotlib.use('Agg')

import json

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from scipy.ndimage import gaussian_filter, zoom

from skimage.metrics import peak_signal_noise_ratio as compute_psnr

from skimage.metrics import structural_similarity as compute_ssim

def evaluate_results(
    ground_truth,
    reconstruction,
    lr_image,
    scale_factor=4,
    elapsed_time=0.0,
    results_dir=None,
    img_size=256,
    lr_size=64,
    noise_std=0.05,
    aa_sigma=1.0,
    tv_weight_stage1=0.08,
    tv_weight_stage2=0.04,
    num_datafid_iters=20
):
    """
    Evaluate reconstruction quality, save metrics and visualizations.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground-truth high-resolution image.
    reconstruction : ndarray
        Reconstructed high-resolution image.
    lr_image : ndarray
        Low-resolution input image.
    scale_factor : int
        Upsampling factor.
    elapsed_time : float
        Total processing time in seconds.
    results_dir : Path or str or None
        Directory to save results.
    img_size : int
        High-resolution image size.
    lr_size : int
        Low-resolution image size.
    noise_std : float
        Noise standard deviation used.
    aa_sigma : float
        Anti-aliasing blur sigma used.
    tv_weight_stage1 : float
        TV weight for stage 1.
    tv_weight_stage2 : float
        TV weight for stage 2.
    num_datafid_iters : int
        Number of data-fidelity iterations.
        
    Returns
    -------
    metrics : dict
        Dictionary containing all evaluation metrics.
    """
    # Compute metrics for DDRM reconstruction
    psnr_val = compute_psnr(ground_truth, reconstruction, data_range=1.0)
    ssim_val = compute_ssim(ground_truth, reconstruction, data_range=1.0)
    rmse_val = np.sqrt(np.mean((ground_truth - reconstruction)**2))
    
    # Compute baseline metrics (bicubic upsampling)
    lr_bicubic = np.clip(zoom(lr_image, scale_factor, order=3), 0, 1)
    psnr_bic = compute_psnr(ground_truth, lr_bicubic, data_range=1.0)
    ssim_bic = compute_ssim(ground_truth, lr_bicubic, data_range=1.0)
    rmse_bic = np.sqrt(np.mean((ground_truth - lr_bicubic)**2))
    
    print(f"[Baseline] Bicubic: PSNR={psnr_bic:.2f} dB, SSIM={ssim_bic:.4f}")
    
    print(f"\n{'=' * 55}")
    print(f"  Results")
    print(f"  {'─' * 50}")
    print(f"  Baseline (Bicubic)  PSNR = {psnr_bic:.4f} dB")
    print(f"  Baseline (Bicubic)  SSIM = {ssim_bic:.4f}")
    print(f"  {'─' * 50}")
    print(f"  DDRM Restoration    PSNR = {psnr_val:.4f} dB")
    print(f"  DDRM Restoration    SSIM = {ssim_val:.4f}")
    print(f"  DDRM Restoration    RMSE = {rmse_val:.4f}")
    print(f"  {'─' * 50}")
    print(f"  Improvement         PSNR = +{psnr_val - psnr_bic:.2f} dB")
    print(f"  Improvement         SSIM = +{ssim_val - ssim_bic:.4f}")
    print(f"  Time                     = {elapsed_time:.2f} s")
    print(f"{'=' * 55}")
    
    metrics = {
        "psnr_db": round(psnr_val, 4),
        "ssim": round(ssim_val, 4),
        "rmse": round(rmse_val, 4),
        "baseline_psnr_db": round(psnr_bic, 4),
        "baseline_ssim": round(ssim_bic, 4),
        "baseline_rmse": round(rmse_bic, 4),
        "method": "DDRM_SVD_restoration",
        "task": "4x_super_resolution",
        "image_size": img_size,
        "lr_size": lr_size,
        "scale_factor": scale_factor,
        "noise_std": noise_std,
        "aa_blur_sigma": aa_sigma,
        "tv_weight_stage1": tv_weight_stage1,
        "tv_weight_stage2": tv_weight_stage2,
        "num_datafid_iters": num_datafid_iters,
        "elapsed_seconds": round(elapsed_time, 2)
    }
    
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[Save] Metrics -> {metrics_path}")
        
        # Save arrays
        gt_path = results_dir / 'ground_truth.npy'
        recon_path = results_dir / 'reconstruction.npy'
        np.save(gt_path, ground_truth)
        np.save(recon_path, reconstruction)
        print(f"[Save] Ground truth -> {gt_path}")
        print(f"[Save] Reconstruction -> {recon_path}")
        
        # Create visualization
        print("[Viz] Creating 4-panel visualization ...")
        error_map = np.abs(ground_truth - reconstruction)
        
        # Upsample LR for display (nearest-neighbor to show pixelation)
        lr_display = np.repeat(np.repeat(lr_image, scale_factor, axis=0),
                               scale_factor, axis=1)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ['Ground Truth',
                  f'Low-Res Input ({scale_factor}x)',
                  'DDRM Reconstruction',
                  'Error Map']
        images = [ground_truth, lr_display, reconstruction, error_map]
        cmaps = ['gray', 'gray', 'gray', 'hot']
        
        for ax, img, title, cmap in zip(axes, images, titles, cmaps):
            vmax = 1.0 if cmap == 'gray' else max(error_map.max(), 0.01)
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(
            f'DDRM SVD-Based Super-Resolution ({scale_factor}x)  |  '
            f'PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  '
            f'RMSE: {rmse_val:.4f}',
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        vis_path = results_dir / 'reconstruction_result.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[Viz] Saved to {vis_path}")
        
        print(f"\n[Done] All outputs saved to {results_dir}")
    
    return metrics
