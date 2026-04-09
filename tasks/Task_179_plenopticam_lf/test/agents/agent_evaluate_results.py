import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

import json

import os

def evaluate_results(gt_scene, gt_depth, recon_center, est_depth, raw_noisy, params, results_dir='results'):
    """
    Compute metrics, generate visualizations, and save results.
    
    Args:
        gt_scene: ground truth scene (2D array)
        gt_depth: ground truth depth map (2D array)
        recon_center: reconstructed center sub-aperture view (2D array)
        est_depth: estimated depth map (2D array)
        raw_noisy: noisy raw MLA image for visualization
        params: dictionary of parameters
        results_dir: directory to save results
    
    Returns:
        dict containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)

    # Compute PSNR for depth
    mse_depth = np.mean((gt_depth - est_depth) ** 2)
    if mse_depth < 1e-15:
        depth_psnr = 100.0
    else:
        data_range_depth = np.max(gt_depth) - np.min(gt_depth)
        if data_range_depth < 1e-15:
            data_range_depth = 1.0
        depth_psnr = 10.0 * np.log10(data_range_depth ** 2 / mse_depth)

    # Compute correlation coefficient for depth
    depth_cc = float(np.corrcoef(gt_depth.ravel(), est_depth.ravel())[0, 1])

    # Compute PSNR for sub-aperture
    mse_sa = np.mean((gt_scene - recon_center) ** 2)
    if mse_sa < 1e-15:
        sa_psnr = 100.0
    else:
        data_range_sa = np.max(gt_scene) - np.min(gt_scene)
        if data_range_sa < 1e-15:
            data_range_sa = 1.0
        sa_psnr = 10.0 * np.log10(data_range_sa ** 2 / mse_sa)

    # Compute SSIM for sub-aperture
    data_range_ssim = max(gt_scene.max() - gt_scene.min(), 
                          recon_center.max() - recon_center.min(), 1e-6)
    sa_ssim = float(ssim(gt_scene, recon_center, data_range=data_range_ssim))

    metrics = {
        "depth_psnr_dB": round(depth_psnr, 2),
        "depth_cc": round(depth_cc, 4),
        "subaperture_psnr_dB": round(sa_psnr, 2),
        "subaperture_ssim": round(sa_ssim, 4),
        "noise_std": params['noise_std'],
        "n_angular": params['n_angular'],
        "scene_size": params['scene_size'],
    }

    # Save metrics to JSON
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    print(f"\n{'='*50}")
    print(f"  Depth  PSNR : {depth_psnr:.2f} dB")
    print(f"  Depth  CC   : {depth_cc:.4f}")
    print(f"  SA     PSNR : {sa_psnr:.2f} dB")
    print(f"  SA     SSIM : {sa_ssim:.4f}")
    print(f"{'='*50}\n")

    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (a) GT center sub-aperture
    ax = axes[0, 0]
    ax.imshow(gt_scene, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(a) GT Center View')
    ax.axis('off')

    # (b) Raw MLA image (small patch)
    ax = axes[0, 1]
    n_angular = params['n_angular']
    patch_size = min(60 * n_angular, raw_noisy.shape[0])
    ax.imshow(raw_noisy[:patch_size, :patch_size], cmap='gray', vmin=0, vmax=1)
    ax.set_title('(b) Raw MLA Image (patch)')
    ax.axis('off')

    # (c) Reconstructed center sub-aperture
    ax = axes[0, 2]
    ax.imshow(recon_center, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(c) Reconstructed Center View')
    ax.axis('off')

    # (d) GT depth map
    ax = axes[1, 0]
    vmin_d, vmax_d = gt_depth.min(), gt_depth.max()
    im_d = ax.imshow(gt_depth, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax.set_title('(d) GT Depth Map')
    ax.axis('off')
    plt.colorbar(im_d, ax=ax, fraction=0.046)

    # (e) Estimated depth map
    ax = axes[1, 1]
    im_e = ax.imshow(est_depth, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax.set_title('(e) Estimated Depth Map')
    ax.axis('off')
    plt.colorbar(im_e, ax=ax, fraction=0.046)

    # (f) Depth error map
    ax = axes[1, 2]
    err = np.abs(gt_depth - est_depth)
    im_f = ax.imshow(err, cmap='hot')
    ax.set_title('(f) Depth Error Map')
    ax.axis('off')
    plt.colorbar(im_f, ax=ax, fraction=0.046)

    plt.suptitle('Task 179: Light Field Reconstruction (plenopticam_lf)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Visualization saved to {save_path}")

    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_depth)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), est_depth)
    print("[DONE] All results saved to results/")

    return metrics
