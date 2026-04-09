import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(ground_truth, reconstruction, sinogram_noisy, n_angles, results_dir=None):
    """
    Evaluate reconstruction results and save outputs.
    
    Parameters:
    -----------
    ground_truth : ndarray
        Ground truth image
    reconstruction : ndarray
        Reconstructed image
    sinogram_noisy : ndarray
        Noisy sinogram input
    n_angles : int
        Number of projection angles (for plot title)
    results_dir : str or None
        Directory to save results. If None, uses ./results
    
    Returns:
    --------
    dict containing:
        - 'psnr_db': Peak Signal-to-Noise Ratio in dB
        - 'ssim': Structural Similarity Index
        - 'cc': Correlation Coefficient
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    gt32 = ground_truth.astype('float32')
    recon = reconstruction.astype('float32')
    
    # Compute metrics
    data_range = float(gt32.max() - gt32.min())
    psnr_val = float(psnr_fn(gt32, recon, data_range=data_range))
    ssim_val = float(ssim_fn(gt32, recon, data_range=data_range))
    cc_val = float(np.corrcoef(gt32.ravel(), recon.ravel())[0, 1])
    
    print(f"\n{'='*50}")
    print(f"  PSNR  = {psnr_val:.2f} dB")
    print(f"  SSIM  = {ssim_val:.4f}")
    print(f"  CC    = {cc_val:.4f}")
    print(f"{'='*50}\n")
    
    # Save numerical results
    metrics = {
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "cc": round(cc_val, 4),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt32)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon)
    np.save(os.path.join(results_dir, "input.npy"), sinogram_noisy.astype('float32'))
    print("[INFO] Saved metrics.json, ground_truth.npy, reconstruction.npy, input.npy")
    
    # Visualization (2×2 panel)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Ground truth
    im0 = axes[0, 0].imshow(gt32, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Ground Truth Phantom", fontsize=13)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # (b) Noisy sinogram
    im1 = axes[0, 1].imshow(sinogram_noisy, cmap='gray', aspect='auto')
    axes[0, 1].set_title("(b) Noisy Sinogram (Input)", fontsize=13)
    axes[0, 1].set_xlabel("Detector pixel")
    axes[0, 1].set_ylabel("Projection angle index")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # (c) TV-PDHG reconstruction
    im2 = axes[1, 0].imshow(recon, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"(c) FBP + TV-PDHG Reconstruction\nPSNR={psnr_val:.1f} dB, SSIM={ssim_val:.3f}",
        fontsize=13,
    )
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # (d) Error map
    error_map = np.abs(gt32 - recon)
    im3 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0,
                             vmax=max(error_map.max(), 0.01))
    axes[1, 1].set_title("(d) Absolute Error |GT − Recon|", fontsize=13)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    fig.suptitle(
        "Task 192: CT Reconstruction via FBP + TV-PDHG\n"
        f"256×256 Shepp-Logan, {n_angles} angles, 1% noise",
        fontsize=15,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved figure → {fig_path}")
    
    return metrics
