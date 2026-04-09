import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_95_eispy2d"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(chi_gt, chi_rec, gx, gy, y_noisy):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters
    ----------
    chi_gt : ndarray
        Ground truth dielectric contrast (n_grid, n_grid)
    chi_rec : ndarray
        Reconstructed dielectric contrast (n_grid, n_grid)
    gx, gy : ndarray
        Grid coordinate vectors
    y_noisy : ndarray
        Noisy scattered field measurements
        
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, SSIM, RMSE values
    """
    # Compute PSNR
    peak = np.max(np.abs(chi_gt))
    if peak == 0:
        psnr_val = 0.0
    else:
        mse = np.mean((chi_gt - chi_rec) ** 2)
        if mse < 1e-30:
            psnr_val = 100.0
        else:
            psnr_val = 10.0 * np.log10(peak ** 2 / mse)
    
    # Compute SSIM
    data_range = max(chi_gt.max() - chi_gt.min(), chi_rec.max() - chi_rec.min(), 1e-10)
    ssim_val = structural_similarity(chi_gt, chi_rec, data_range=data_range)
    
    # Compute RMSE
    rmse_val = float(np.sqrt(np.mean((chi_gt - chi_rec) ** 2)))
    
    metrics = {"PSNR": psnr_val, "SSIM": ssim_val, "RMSE": rmse_val}
    
    print(f"  PSNR = {psnr_val:.2f} dB")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  RMSE = {rmse_val:.6f}")
    
    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), chi_gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), chi_rec)
    np.save(os.path.join(RESULTS_DIR, "scattered_field.npy"), y_noisy)
    
    # Website assets
    np.save(os.path.join(ASSETS_DIR, "gt_output.npy"), chi_gt)
    np.save(os.path.join(ASSETS_DIR, "recon_output.npy"), chi_rec)
    
    # Metrics JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    extent = [gx[0] * 1e3, gx[-1] * 1e3, gy[0] * 1e3, gy[-1] * 1e3]  # mm
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    im0 = axes[0].imshow(chi_gt, extent=extent, origin="lower",
                         cmap="jet", vmin=0, vmax=1.1)
    axes[0].set_title("Ground Truth  χ(r)")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)
    
    im1 = axes[1].imshow(chi_rec, extent=extent, origin="lower",
                         cmap="jet", vmin=0, vmax=1.1)
    axes[1].set_title("Reconstructed  χ̂(r)")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)
    
    diff = np.abs(chi_gt - chi_rec)
    im2 = axes[2].imshow(diff, extent=extent, origin="lower", cmap="hot")
    axes[2].set_title("|Error|")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("y [mm]")
    plt.colorbar(im2, ax=axes[2], shrink=0.85)
    
    fig.suptitle(
        f"EM Inverse Scattering (Born + Tikhonov)   "
        f"PSNR={metrics['PSNR']:.2f} dB   SSIM={metrics['SSIM']:.4f}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    
    # Save plots
    vis_paths = [
        os.path.join(RESULTS_DIR, "vis_result.png"),
        os.path.join(ASSETS_DIR, "vis_result.png"),
        os.path.join(WORKING_DIR, "vis_result.png"),
    ]
    for p in vis_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)
    
    return metrics
