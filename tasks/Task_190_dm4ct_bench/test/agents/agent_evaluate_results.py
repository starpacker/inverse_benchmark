import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

def evaluate_results(phantom, recon_fbp, recon_diffusion, sino_noisy, results_dir, 
                     n_angles_sparse, n_angles_full, noise_level, n_outer_iter):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters:
    -----------
    phantom : ndarray
        Ground truth phantom image
    recon_fbp : ndarray
        FBP baseline reconstruction
    recon_diffusion : ndarray
        Diffusion-style reconstruction
    sino_noisy : ndarray
        Noisy sparse-view sinogram (for visualization)
    results_dir : str
        Directory to save results
    n_angles_sparse : int
        Number of sparse angles (for metadata)
    n_angles_full : int
        Number of full angles (for metadata)
    noise_level : float
        Noise level (for metadata)
    n_outer_iter : int
        Number of iterations (for metadata)
        
    Returns:
    --------
    metrics_diff : dict
        Dictionary containing PSNR, SSIM, RMSE for diffusion reconstruction
    """
    # Metric computation helpers
    def compute_psnr(ref, test, data_range=None):
        """Compute PSNR (dB)."""
        if data_range is None:
            data_range = ref.max() - ref.min()
        mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)
    
    def compute_ssim(ref, test):
        """Compute SSIM."""
        data_range = ref.max() - ref.min()
        if data_range == 0:
            data_range = 1.0
        return ssim(ref, test, data_range=data_range)
    
    def compute_rmse(ref, test):
        """Compute RMSE."""
        return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))
    
    # Compute metrics
    metrics_fbp = {
        "psnr": float(compute_psnr(phantom, np.clip(recon_fbp, 0, 1))),
        "ssim": float(compute_ssim(phantom, np.clip(recon_fbp, 0, 1))),
        "rmse": float(compute_rmse(phantom, np.clip(recon_fbp, 0, 1))),
    }
    metrics_diff = {
        "psnr": float(compute_psnr(phantom, np.clip(recon_diffusion, 0, 1))),
        "ssim": float(compute_ssim(phantom, np.clip(recon_diffusion, 0, 1))),
        "rmse": float(compute_rmse(phantom, np.clip(recon_diffusion, 0, 1))),
    }
    
    print(f"\n[EVAL] FBP Baseline:  PSNR={metrics_fbp['psnr']:.2f}dB, SSIM={metrics_fbp['ssim']:.4f}")
    print(f"[EVAL] Diffusion-CT:  PSNR={metrics_diff['psnr']:.2f}dB, SSIM={metrics_diff['ssim']:.4f}")
    print(f"[EVAL] Improvement:   ΔPSNR={metrics_diff['psnr']-metrics_fbp['psnr']:+.2f}dB, "
          f"ΔSSIM={metrics_diff['ssim']-metrics_fbp['ssim']:+.4f}")
    
    # Save metrics
    metrics = {
        "psnr": metrics_diff["psnr"],
        "ssim": metrics_diff["ssim"],
        "rmse": metrics_diff["rmse"],
        "fbp_psnr": metrics_fbp["psnr"],
        "fbp_ssim": metrics_fbp["ssim"],
        "n_angles_sparse": n_angles_sparse,
        "n_angles_full": n_angles_full,
        "noise_level": noise_level,
        "n_iterations": n_outer_iter,
        "method": "Diffusion-style iterative refinement (TV prior + data consistency)",
    }
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")
    
    # Generate visualization
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Panel 1: Ground Truth
    axes[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    # Panel 2: Sparse Sinogram
    axes[1].imshow(sino_noisy, cmap='gray', aspect='auto')
    axes[1].set_title(f'Sparse Sinogram\n({n_angles_sparse} views)', fontsize=12)
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('Detector')
    
    # Panel 3: FBP Baseline
    axes[2].imshow(np.clip(recon_fbp, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'FBP Baseline\nPSNR={metrics_fbp["psnr"]:.1f}dB', fontsize=12)
    axes[2].axis('off')
    
    # Panel 4: Diffusion-style Reconstruction
    axes[3].imshow(np.clip(recon_diffusion, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'Iterative Recon\nPSNR={metrics_diff["psnr"]:.1f}dB', fontsize=12)
    axes[3].axis('off')
    
    # Panel 5: Error Map
    error = np.abs(phantom - recon_diffusion)
    axes[4].imshow(error, cmap='hot', vmin=0, vmax=0.3)
    axes[4].set_title('|Error|', fontsize=12)
    axes[4].axis('off')
    
    fig.suptitle(
        f"DM4CT Sparse-View CT | Diffusion-Style: PSNR={metrics_diff['psnr']:.2f}dB, "
        f"SSIM={metrics_diff['ssim']:.4f} | FBP: PSNR={metrics_fbp['psnr']:.2f}dB",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_diffusion)
    np.save(os.path.join(results_dir, "ground_truth.npy"), phantom)
    
    return metrics_diff
