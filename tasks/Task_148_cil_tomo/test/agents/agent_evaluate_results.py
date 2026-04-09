import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def evaluate_results(phantom, recon_fbp, recon_cgls, recon_tv, sinogram_noisy,
                     n_angles, noise_level):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        phantom: Ground truth image
        recon_fbp: FBP reconstruction
        recon_cgls: CGLS reconstruction
        recon_tv: TV-FISTA reconstruction
        sinogram_noisy: Noisy sinogram (for visualization)
        n_angles: Number of projection angles
        noise_level: Noise level used
        
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    
    def compute_metrics(gt, recon):
        """Compute PSNR, SSIM, RMSE between ground truth and reconstruction."""
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
        recon_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)
        
        psnr = peak_signal_noise_ratio(gt_n, recon_n, data_range=1.0)
        ssim = structural_similarity(gt_n, recon_n, data_range=1.0)
        rmse = np.sqrt(np.mean((gt_n - recon_n) ** 2))
        return psnr, ssim, rmse
    
    def norm01(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    # Compute metrics for each method
    psnr_fbp, ssim_fbp, rmse_fbp = compute_metrics(phantom, recon_fbp)
    print(f"\n  FBP  — PSNR: {psnr_fbp:.2f} dB, "
          f"SSIM: {ssim_fbp:.4f}, RMSE: {rmse_fbp:.4f}")
    
    psnr_cgls, ssim_cgls, rmse_cgls = compute_metrics(phantom, recon_cgls)
    print(f"  CGLS — PSNR: {psnr_cgls:.2f} dB, "
          f"SSIM: {ssim_cgls:.4f}, RMSE: {rmse_cgls:.4f}")
    
    psnr_tv, ssim_tv, rmse_tv = compute_metrics(phantom, recon_tv)
    print(f"  TV   — PSNR: {psnr_tv:.2f} dB, "
          f"SSIM: {ssim_tv:.4f}, RMSE: {rmse_tv:.4f}")
    
    # Determine best method
    results = {
        'FBP':      (recon_fbp,  psnr_fbp,  ssim_fbp,  rmse_fbp),
        'CGLS':     (recon_cgls, psnr_cgls, ssim_cgls, rmse_cgls),
        'TV-FISTA': (recon_tv,   psnr_tv,   ssim_tv,   rmse_tv),
    }
    best_name = max(results, key=lambda k: results[k][1])
    best_recon, best_psnr, best_ssim, best_rmse = results[best_name]
    
    print(f"\n★ Best method: {best_name}")
    print(f"  PSNR: {best_psnr:.2f} dB | "
          f"SSIM: {best_ssim:.4f} | RMSE: {best_rmse:.4f}")
    
    # Save outputs
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), phantom)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_recon)
    
    metrics = {
        "PSNR": round(float(best_psnr), 2),
        "SSIM": round(float(best_ssim), 4),
        "RMSE": round(float(best_rmse), 4),
        "best_method": best_name,
        "n_angles": n_angles,
        "noise_level": noise_level,
        "FBP": {
            "PSNR": round(float(psnr_fbp), 2),
            "SSIM": round(float(ssim_fbp), 4),
            "RMSE": round(float(rmse_fbp), 4),
        },
        "CGLS": {
            "PSNR": round(float(psnr_cgls), 2),
            "SSIM": round(float(ssim_cgls), 4),
            "RMSE": round(float(rmse_cgls), 4),
        },
        "TV-FISTA": {
            "PSNR": round(float(psnr_tv), 2),
            "SSIM": round(float(ssim_tv), 4),
            "RMSE": round(float(rmse_tv), 4),
        },
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.json")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: GT, Sinogram, FBP
    axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Ground Truth\n(Shepp-Logan 128×128)", fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sinogram_noisy, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f"Sinogram (noisy)\n{n_angles} projections, "
                         f"σ={noise_level}", fontsize=11)
    axes[0, 1].set_xlabel("Detector position")
    axes[0, 1].set_ylabel("Projection angle")
    
    axes[0, 2].imshow(norm01(recon_fbp), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f"FBP Reconstruction\n"
                         f"PSNR={psnr_fbp:.1f} dB, "
                         f"SSIM={ssim_fbp:.3f}", fontsize=11)
    axes[0, 2].axis('off')
    
    # Row 2: CGLS, TV-FISTA, Error map
    axes[1, 0].imshow(norm01(recon_cgls), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"CGLS Reconstruction\n"
                         f"PSNR={psnr_cgls:.1f} dB, "
                         f"SSIM={ssim_cgls:.3f}", fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(norm01(recon_tv), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f"TV-FISTA Reconstruction\n"
                         f"PSNR={psnr_tv:.1f} dB, "
                         f"SSIM={ssim_tv:.3f}", fontsize=11)
    axes[1, 1].axis('off')
    
    error_map = np.abs(phantom - norm01(best_recon))
    im = axes[1, 2].imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title(f"Error Map ({best_name})\n"
                         f"RMSE={best_rmse:.4f}", fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    fig.suptitle(
        "CT Tomographic Reconstruction — Iterative Optimization\n"
        "(Radon forward model · FBP vs CGLS vs TV-FISTA)",
        fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    
    return metrics
