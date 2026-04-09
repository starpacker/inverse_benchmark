import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

def evaluate_results(gt, recon, sinogram, results_dir, assets_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes metrics:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - RMSE: Root Mean Square Error
    - CC: Pearson Correlation Coefficient
    
    Also generates visualization and saves all outputs.
    
    Parameters
    ----------
    gt : ndarray
        Ground truth attenuation map
    recon : ndarray
        Reconstructed attenuation map
    sinogram : ndarray
        Sinogram used for reconstruction
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
        
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, SSIM, RMSE, CC values
    """
    # Crop to same size (iradon may produce slightly different shape)
    min_h = min(gt.shape[0], recon.shape[0])
    min_w = min(gt.shape[1], recon.shape[1])
    gt_c = gt[:min_h, :min_w]
    re_c = recon[:min_h, :min_w]

    # RMSE
    rmse = np.sqrt(np.mean((gt_c - re_c)**2))

    # Data range
    data_range = gt_c.max() - gt_c.min()
    if data_range < 1e-10:
        data_range = 1.0

    # PSNR
    mse = np.mean((gt_c - re_c)**2)
    psnr = 10 * np.log10(data_range**2 / (mse + 1e-12))

    # SSIM
    ssim_val = ssim(gt_c, re_c, data_range=data_range)

    # CC (Pearson correlation)
    g = gt_c.flatten() - gt_c.mean()
    r = re_c.flatten() - re_c.mean()
    cc = np.sum(g * r) / (np.sqrt(np.sum(g**2) * np.sum(r**2)) + 1e-12)

    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "RMSE": float(rmse),
        "CC": float(cc),
    }

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    im0 = axes[0, 0].imshow(sinogram.T, cmap="gray", aspect="auto",
                             extent=[0, 180, -sinogram.shape[0]//2, sinogram.shape[0]//2])
    axes[0, 0].set_title("Sinogram (neutron transmission)", fontsize=14)
    axes[0, 0].set_xlabel("Angle (degrees)")
    axes[0, 0].set_ylabel("Detector position")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(gt_n, cmap="inferno")
    axes[0, 1].set_title("Ground Truth (μ distribution)", fontsize=14)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(re_n, cmap="inferno")
    axes[1, 0].set_title(
        f"FBP Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"SSIM={metrics['SSIM']:.4f}",
        fontsize=12,
    )
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Crop for error visualization
    min_h_vis = min(gt_n.shape[0], re_n.shape[0])
    min_w_vis = min(gt_n.shape[1], re_n.shape[1])
    error = np.abs(gt_n[:min_h_vis, :min_w_vis] - re_n[:min_h_vis, :min_w_vis])
    im3 = axes[1, 1].imshow(error, cmap="magma")
    axes[1, 1].set_title(f"Absolute Error (RMSE={metrics['RMSE']:.4f} cm⁻¹)", fontsize=12)
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    
    # Save figures
    for p in [os.path.join(results_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()

    # Save data
    for d in [results_dir, assets_dir]:
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "gt_output.npy"), gt)
        np.save(os.path.join(d, "recon_output.npy"), recon)
        np.save(os.path.join(d, "ground_truth.npy"), gt)
        np.save(os.path.join(d, "reconstruction.npy"), recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
