import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

def evaluate_results(phantom, tv_recon, fbp_sparse, fbp_full, config, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        phantom: Ground truth image
        tv_recon: TV-regularized reconstruction
        fbp_sparse: FBP reconstruction from sparse data
        fbp_full: FBP reconstruction from full data
        config: Configuration dictionary
        results_dir: Directory to save results
        
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    
    def compute_psnr(gt, recon, data_range=None):
        """Compute PSNR between ground truth and reconstruction."""
        if data_range is None:
            data_range = gt.max() - gt.min()
        mse = np.mean((gt - recon) ** 2)
        if mse < 1e-12:
            return 100.0
        return 10.0 * np.log10(data_range ** 2 / mse)
    
    def compute_rmse(gt, recon):
        """Compute RMSE between ground truth and reconstruction."""
        return np.sqrt(np.mean((gt - recon) ** 2))
    
    def compute_ssim_metric(gt, recon):
        """Compute SSIM between ground truth and reconstruction."""
        data_range = gt.max() - gt.min()
        return ssim(gt, recon, data_range=data_range)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Clip reconstructions to [0, 1]
    tv_recon_clipped = np.clip(tv_recon, 0, 1)
    fbp_sparse_clipped = np.clip(fbp_sparse, 0, 1)
    fbp_full_clipped = np.clip(fbp_full, 0, 1)
    
    # Compute metrics for FBP full
    psnr_fbp_full = compute_psnr(phantom, fbp_full_clipped, data_range=1.0)
    ssim_fbp_full = compute_ssim_metric(phantom, fbp_full_clipped)
    
    # Compute metrics for FBP sparse
    psnr_fbp_sparse = compute_psnr(phantom, fbp_sparse_clipped, data_range=1.0)
    ssim_fbp_sparse = compute_ssim_metric(phantom, fbp_sparse_clipped)
    
    # Compute metrics for TV reconstruction
    psnr_tv = compute_psnr(phantom, tv_recon_clipped, data_range=1.0)
    ssim_tv = compute_ssim_metric(phantom, tv_recon_clipped)
    rmse_tv = compute_rmse(phantom, tv_recon_clipped)
    
    print(f"\n  FBP full ({config['n_full_angles']} angles): PSNR={psnr_fbp_full:.2f}dB, SSIM={ssim_fbp_full:.4f}")
    print(f"  FBP sparse ({config['n_sparse_angles']} angles): PSNR={psnr_fbp_sparse:.2f}dB, SSIM={ssim_fbp_sparse:.4f}")
    print(f"\n  FISTA-TV result: PSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.4f}, RMSE={rmse_tv:.4f}")
    print(f"  Improvement over sparse FBP: "
          f"PSNR +{psnr_tv - psnr_fbp_sparse:.2f}dB, "
          f"SSIM +{ssim_tv - ssim_fbp_sparse:.4f}")
    
    # Save numpy arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), phantom)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), tv_recon_clipped)
    np.save(os.path.join(results_dir, 'fbp_sparse.npy'), fbp_sparse_clipped)
    print("  Saved .npy files")
    
    # Create metrics dictionary
    metrics = {
        "task": "saxnerf_ct",
        "description": "Sparse-view CT reconstruction using FISTA-TV",
        "phantom_size": config['size'],
        "n_full_angles": config['n_full_angles'],
        "n_sparse_angles": config['n_sparse_angles'],
        "fbp_full": {
            "psnr_db": round(psnr_fbp_full, 4),
            "ssim": round(ssim_fbp_full, 4)
        },
        "fbp_sparse": {
            "psnr_db": round(psnr_fbp_sparse, 4),
            "ssim": round(ssim_fbp_sparse, 4)
        },
        "fista_tv": {
            "psnr_db": round(psnr_tv, 4),
            "ssim": round(ssim_tv, 4),
            "rmse": round(rmse_tv, 6),
            "n_iterations": 200,
            "tv_weight": 0.008
        },
        "improvement_over_sparse_fbp": {
            "psnr_gain_db": round(psnr_tv - psnr_fbp_sparse, 4),
            "ssim_gain": round(ssim_tv - ssim_fbp_sparse, 4)
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Ground Truth
    im0 = axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Sparse FBP
    im1 = axes[0, 1].imshow(fbp_sparse_clipped, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Sparse FBP ({config["n_sparse_angles"]} angles)\nPSNR={psnr_fbp_sparse:.1f}dB, SSIM={ssim_fbp_sparse:.3f}',
                         fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # TV Reconstruction
    im2 = axes[1, 0].imshow(tv_recon_clipped, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'FISTA-TV Recon ({config["n_sparse_angles"]} angles)\nPSNR={psnr_tv:.1f}dB, SSIM={ssim_tv:.3f}',
                         fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Error map (TV recon)
    error_map = np.abs(phantom - tv_recon_clipped)
    im3 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title(f'Error Map (|GT - TV Recon|)\nRMSE={rmse_tv:.4f}',
                         fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.suptitle('Sparse-View CT Reconstruction\n(SAX-NeRF Task: sparse projections → CT volume)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ground truth: Shepp-Logan phantom {config['size']}x{config['size']}")
    print(f"  Sparse angles: {config['n_sparse_angles']} (of {config['n_full_angles']})")
    print(f"  FBP sparse:  PSNR={psnr_fbp_sparse:.2f}dB, SSIM={ssim_fbp_sparse:.4f}")
    print(f"  FISTA-TV:    PSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.4f}, RMSE={rmse_tv:.6f}")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)
    
    return metrics
