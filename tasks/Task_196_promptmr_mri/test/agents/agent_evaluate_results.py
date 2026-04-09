import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

def compute_psnr(gt, recon):
    """Compute PSNR."""
    mse = np.mean((gt - recon)**2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10.0 * np.log10(data_range**2 / mse)

def compute_ssim(gt, recon):
    """Compute SSIM."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)

def evaluate_results(data_dict, result_dict, output_dir='results'):
    """
    Evaluate reconstruction quality and save results.
    
    Parameters:
    -----------
    data_dict : dict
        Contains ground truth images and parameters
    result_dict : dict
        Contains reconstruction results
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all evaluation metrics
    """
    print("\n[4/4] Evaluating metrics...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    t1_gt = data_dict['t1_gt']
    t2_gt = data_dict['t2_gt']
    N = data_dict['N']
    t1_acceleration = data_dict['t1_acceleration']
    t2_acceleration = data_dict['t2_acceleration']
    
    recon_t1 = result_dict['recon_t1']
    recon_t2 = result_dict['recon_t2']
    zf_t1 = result_dict['zf_t1']
    zf_t2 = result_dict['zf_t2']
    n_iter = result_dict['n_iter']
    
    # Compute metrics for zero-filled
    psnr_zf_t1 = compute_psnr(t1_gt, zf_t1)
    psnr_zf_t2 = compute_psnr(t2_gt, zf_t2)
    ssim_zf_t1 = compute_ssim(t1_gt, zf_t1)
    ssim_zf_t2 = compute_ssim(t2_gt, zf_t2)
    
    # Compute metrics for reconstruction
    psnr_t1 = compute_psnr(t1_gt, recon_t1)
    psnr_t2 = compute_psnr(t2_gt, recon_t2)
    ssim_t1 = compute_ssim(t1_gt, recon_t1)
    ssim_t2 = compute_ssim(t2_gt, recon_t2)
    
    psnr_avg = (psnr_t1 + psnr_t2) / 2.0
    ssim_avg = (ssim_t1 + ssim_t2) / 2.0
    
    print(f"\n  Zero-filled baselines:")
    print(f"    T1: PSNR={psnr_zf_t1:.2f} dB, SSIM={ssim_zf_t1:.4f}")
    print(f"    T2: PSNR={psnr_zf_t2:.2f} dB, SSIM={ssim_zf_t2:.4f}")
    print(f"\n  CS-TV Reconstruction:")
    print(f"    T1 ({t1_acceleration}x): PSNR={psnr_t1:.2f} dB, SSIM={ssim_t1:.4f}")
    print(f"    T2 ({t2_acceleration}x): PSNR={psnr_t2:.2f} dB, SSIM={ssim_t2:.4f}")
    print(f"    Average: PSNR={psnr_avg:.2f} dB, SSIM={ssim_avg:.4f}")
    
    # Save metrics
    metrics = {
        "task": "promptmr_mri",
        "method": "FISTA-TV CS-MRI Reconstruction",
        "psnr_t1": round(psnr_t1, 2),
        "ssim_t1": round(ssim_t1, 4),
        "psnr_t2": round(psnr_t2, 2),
        "ssim_t2": round(ssim_t2, 4),
        "psnr_avg": round(psnr_avg, 2),
        "ssim_avg": round(ssim_avg, 4),
        "psnr_zf_t1": round(psnr_zf_t1, 2),
        "psnr_zf_t2": round(psnr_zf_t2, 2),
        "ssim_zf_t1": round(ssim_zf_t1, 4),
        "ssim_zf_t2": round(ssim_zf_t2, 4),
        "t1_acceleration": t1_acceleration,
        "t2_acceleration": t2_acceleration,
        "image_size": N,
        "fista_iterations": n_iter,
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {output_dir}/metrics.json")
    
    # Save arrays
    gt_stack = np.stack([t1_gt, t2_gt], axis=0)
    recon_stack = np.stack([recon_t1, recon_t2], axis=0)
    np.save(os.path.join(output_dir, 'ground_truth.npy'), gt_stack)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_stack)
    print(f"  Arrays saved: ground_truth.npy {gt_stack.shape}, reconstruction.npy {recon_stack.shape}")
    
    # Visualization
    print("\n  Generating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    vmax_err = 0.15
    
    # Row 1: T1
    axes[0, 0].imshow(t1_gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('T1 Ground Truth', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(zf_t1, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'T1 Zero-filled ({t1_acceleration}x)\nPSNR={psnr_zf_t1:.1f}dB', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(recon_t1, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'T1 CS-TV Recon\nPSNR={psnr_t1:.1f}dB, SSIM={ssim_t1:.3f}', fontsize=12)
    axes[0, 2].axis('off')
    
    err_t1 = np.abs(t1_gt - recon_t1)
    im1 = axes[0, 3].imshow(err_t1, cmap='hot', vmin=0, vmax=vmax_err)
    axes[0, 3].set_title('T1 Error (×5)', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: T2
    axes[1, 0].imshow(t2_gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('T2 Ground Truth', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(zf_t2, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'T2 Zero-filled ({t2_acceleration}x)\nPSNR={psnr_zf_t2:.1f}dB', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(recon_t2, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'T2 CS-TV Recon\nPSNR={psnr_t2:.1f}dB, SSIM={ssim_t2:.3f}', fontsize=12)
    axes[1, 2].axis('off')
    
    err_t2 = np.abs(t2_gt - recon_t2)
    im2 = axes[1, 3].imshow(err_t2, cmap='hot', vmin=0, vmax=vmax_err)
    axes[1, 3].set_title('T2 Error (×5)', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)
    
    fig.suptitle(
        f'Multi-Contrast MRI Reconstruction (PromptMR-style)\n'
        f'T1({t1_acceleration}x): PSNR={psnr_t1:.2f}dB/SSIM={ssim_t1:.4f}  |  '
        f'T2({t2_acceleration}x): PSNR={psnr_t2:.2f}dB/SSIM={ssim_t2:.4f}  |  '
        f'Avg: PSNR={psnr_avg:.2f}dB/SSIM={ssim_avg:.4f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {output_dir}/reconstruction_result.png")
    
    print("\n" + "=" * 60)
    print(f"DONE — Average PSNR: {psnr_avg:.2f} dB, SSIM: {ssim_avg:.4f}")
    print("=" * 60)
    
    return metrics
