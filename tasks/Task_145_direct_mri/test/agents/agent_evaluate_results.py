import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(gt_norm, zf_norm, recon_norm, recon_metrics, method_name, 
                     zf_metrics, N, acceleration, results_dir):
    """
    Evaluate reconstruction results, create visualizations, and save outputs.
    
    Args:
        gt_norm: Normalized ground truth image
        zf_norm: Normalized zero-filled reconstruction
        recon_norm: Best reconstruction result
        recon_metrics: (PSNR, SSIM, RMSE) for reconstruction
        method_name: Name of the reconstruction method
        zf_metrics: (PSNR, SSIM, RMSE) for zero-filled baseline
        N: Image size
        acceleration: Acceleration factor
        results_dir: Directory to save results
    
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute error map
    error_map = np.abs(gt_norm - recon_norm)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(gt_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(zf_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Zero-Filled IFFT\nPSNR={zf_metrics[0]:.1f}dB, SSIM={zf_metrics[1]:.3f}', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'{method_name}\nPSNR={recon_metrics[0]:.1f}dB, SSIM={recon_metrics[1]:.3f}', fontsize=12)
    axes[2].axis('off')

    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.15)
    axes[3].set_title('Error Map (|GT - Recon|)', fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('Task 145: Deep Learning MRI Reconstruction', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_norm)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_norm)
    
    # Create metrics dictionary
    metrics_dict = {
        'task': 'direct_mri',
        'method': method_name,
        'PSNR': float(round(recon_metrics[0], 4)),
        'SSIM': float(round(recon_metrics[1], 4)),
        'RMSE': float(round(recon_metrics[2], 4)),
        'zero_filled_PSNR': float(round(zf_metrics[0], 4)),
        'zero_filled_SSIM': float(round(zf_metrics[1], 4)),
        'image_size': N,
        'acceleration': acceleration,
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved metrics.json")
    
    # Print summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print(f"  PSNR  : {recon_metrics[0]:.2f} dB {'PASS' if recon_metrics[0] > 15 else 'FAIL'}")
    print(f"  SSIM  : {recon_metrics[1]:.4f} {'PASS' if recon_metrics[1] > 0.5 else 'FAIL'}")
    print(f"  RMSE  : {recon_metrics[2]:.4f}")
    print(f"  Method: {method_name}")
    status = "PASS" if recon_metrics[0] > 15 and recon_metrics[1] > 0.5 else "FAIL"
    print(f"  Status: {status}")
    print("=" * 65)
    
    return metrics_dict
