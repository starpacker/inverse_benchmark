import os

import json

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def evaluate_results(ground_truth, reconstructions_dict, results_dir):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR, SSIM, and RMSE for each reconstruction method,
    generates visualization, and saves metrics to files.
    
    Args:
        ground_truth: Ground truth image (2D numpy array)
        reconstructions_dict: Dictionary mapping method names to tuples of 
                             (reconstruction, elapsed_time, sinogram_noisy, 
                              theta, actual_snr, image_size, num_angles)
        results_dir: Directory to save results
        
    Returns:
        results: Dictionary of metrics for all methods
        best_method: Name of the best performing method
        best_metrics: Metrics dictionary for the best method
    """
    os.makedirs(results_dir, exist_ok=True)
    
    def compute_metrics(gt, rec):
        """Compute PSNR, SSIM, and RMSE."""
        gt_copy = gt.copy()
        rec_copy = np.clip(rec.copy(), 0, None)
        
        gt_max = gt_copy.max()
        if gt_max > 0:
            gt_norm = gt_copy / gt_max
            rec_norm = rec_copy / gt_max
            rec_norm = np.clip(rec_norm, 0, 1)
        else:
            gt_norm = gt_copy
            rec_norm = rec_copy
        
        psnr = peak_signal_noise_ratio(gt_norm, rec_norm, data_range=1.0)
        ssim = structural_similarity(gt_norm, rec_norm, data_range=1.0)
        rmse = np.sqrt(mean_squared_error(gt_norm, rec_norm))
        return psnr, ssim, rmse
    
    results = {}
    recons = {}
    
    # Extract common parameters from first entry
    first_key = list(reconstructions_dict.keys())[0]
    _, _, sinogram_noisy, theta, actual_snr, image_size, num_angles = reconstructions_dict[first_key]
    
    # Evaluate each method
    for method_name, (recon, elapsed_time, _, _, _, _, _) in reconstructions_dict.items():
        psnr, ssim, rmse = compute_metrics(ground_truth, recon)
        results[method_name] = {
            'psnr': psnr,
            'ssim': ssim,
            'rmse': rmse,
            'time': elapsed_time
        }
        recons[method_name] = recon
        print(f"    {method_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
              f"RMSE={rmse:.4f}, time={elapsed_time:.2f}s")
    
    # Find best methods
    best_method = max(results, key=lambda k: results[k]['psnr'])
    best_recon = recons[best_method]
    best_metrics = results[best_method]
    
    # Best iterative method
    iter_keys = [k for k in results if 'SIRT' in k]
    if iter_keys:
        best_iter = max(iter_keys, key=lambda k: results[k]['psnr'])
        best_iter_recon = recons[best_iter]
        best_iter_metrics = results[best_iter]
    else:
        best_iter = best_method
        best_iter_recon = best_recon
        best_iter_metrics = best_metrics
    
    # Best FBP method
    fbp_keys = [k for k in results if k.startswith('FBP')]
    if fbp_keys:
        best_fbp = max(fbp_keys, key=lambda k: results[k]['psnr'])
        best_fbp_recon = recons[best_fbp]
        best_fbp_metrics = results[best_fbp]
    else:
        best_fbp = best_method
        best_fbp_recon = best_recon
        best_fbp_metrics = best_metrics
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(ground_truth, cmap='gray', vmin=0, vmax=ground_truth.max())
    ax.set_title(f'(a) Ground Truth\n(Shepp-Logan Phantom {image_size}×{image_size})',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (b) Sinogram
    ax = axes[0, 1]
    im = ax.imshow(sinogram_noisy, cmap='hot', aspect='auto',
                   extent=[theta[0], theta[-1], sinogram_noisy.shape[0], 0])
    ax.set_title(f'(b) Noisy Sinogram\nSNR={actual_snr:.1f} dB, {num_angles} angles',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Detector position')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (c) Best FBP reconstruction
    ax = axes[1, 0]
    disp = np.clip(best_fbp_recon, 0, ground_truth.max())
    im = ax.imshow(disp, cmap='gray', vmin=0, vmax=ground_truth.max())
    ax.set_title(f'(c) {best_fbp} Reconstruction\nPSNR={best_fbp_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_fbp_metrics["ssim"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (d) Best iterative reconstruction
    ax = axes[1, 1]
    disp = np.clip(best_iter_recon, 0, ground_truth.max())
    im = ax.imshow(disp, cmap='gray', vmin=0, vmax=ground_truth.max())
    ax.set_title(f'(d) {best_iter} Reconstruction\nPSNR={best_iter_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_iter_metrics["ssim"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle('CT Reconstruction via Radon Transform Inversion\n'
                 f'Best: {best_method} — PSNR={best_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_metrics["ssim"]:.4f}',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved figure: {fig_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), ground_truth)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_recon)
    
    # Save metrics JSON
    metrics_output = {
        "task": "torch_radon_ct",
        "inverse_problem": "CT reconstruction via Radon transform inversion",
        "image_size": image_size,
        "num_angles": num_angles,
        "noise_snr_db": float(actual_snr),
        "best_method": best_method,
        "best_psnr_db": float(best_metrics['psnr']),
        "best_ssim": float(best_metrics['ssim']),
        "best_rmse": float(best_metrics['rmse']),
        "all_methods": {
            method: {
                "psnr_db": float(v['psnr']),
                "ssim": float(v['ssim']),
                "rmse": float(v['rmse']),
                "time_seconds": float(v['time'])
            } for method, v in results.items()
        }
    }
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print("    Saved: ground_truth.npy, reconstruction.npy, metrics.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ground truth:   {image_size}x{image_size} Shepp-Logan phantom")
    print(f"  Forward model:  Radon transform, {num_angles} angles, SNR={actual_snr:.1f} dB")
    print(f"  Methods compared:")
    for method, v in results.items():
        print(f"    {method:14s}  PSNR={v['psnr']:.2f} dB  SSIM={v['ssim']:.4f}  "
              f"RMSE={v['rmse']:.4f}  t={v['time']:.1f}s")
    print(f"  Best method:    {best_method}")
    print(f"  Best PSNR:      {best_metrics['psnr']:.2f} dB")
    print(f"  Best SSIM:      {best_metrics['ssim']:.4f}")
    print("=" * 60)
    
    return results, best_method, best_metrics
