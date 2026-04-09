import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_116_thermoelastic"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(gt_stress_sum, recon_stress_sum, R, THETA, delta_T_noisy):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Computes PSNR, SSIM, RMSE, and correlation coefficient metrics,
    saves results to disk, and creates visualization plots.
    
    Parameters
    ----------
    gt_stress_sum : ndarray
        Ground truth stress sum field (Pa)
    recon_stress_sum : ndarray
        Reconstructed stress sum field (Pa)
    R : ndarray
        Radial coordinate meshgrid
    THETA : ndarray
        Angular coordinate meshgrid
    delta_T_noisy : ndarray
        Noisy temperature measurements (K)
    
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, CC, RMSE, RMSE_MPa
    """
    # Compute metrics
    mse = np.mean((gt_stress_sum - recon_stress_sum) ** 2)
    data_range = np.max(gt_stress_sum) - np.min(gt_stress_sum)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))
    
    cc = float(np.corrcoef(gt_stress_sum.ravel(), recon_stress_sum.ravel())[0, 1])
    rmse = float(np.sqrt(mse))
    
    # SSIM — normalize to [0, 1]
    gt_n = (gt_stress_sum - gt_stress_sum.min()) / (gt_stress_sum.max() - gt_stress_sum.min() + 1e-30)
    rc_n = (recon_stress_sum - recon_stress_sum.min()) / (recon_stress_sum.max() - recon_stress_sum.min() + 1e-30)
    ssim_val = float(ssim(gt_n, rc_n, data_range=1.0))
    
    metrics = {
        "PSNR": float(psnr),
        "SSIM": ssim_val,
        "CC": cc,
        "RMSE": rmse,
        "RMSE_MPa": rmse / 1e6
    }
    
    # Print metrics
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  SSIM = {metrics['SSIM']:.4f}")
    print(f"  CC   = {metrics['CC']:.6f}")
    print(f"  RMSE = {metrics['RMSE_MPa']:.4f} MPa")
    
    # Save numerical results
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_stress_sum)
        np.save(os.path.join(d, "recon_output.npy"), recon_stress_sum)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # (a) GT stress sum
    ax = axes[0, 0]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, gt_stress_sum / 1e6, cmap="RdBu_r", shading="auto")
    ax.set_title("GT Stress Sum  σ₁+σ₂  (MPa)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # (b) Temperature change
    ax = axes[0, 1]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, delta_T_noisy * 1e3, cmap="coolwarm", shading="auto")
    ax.set_title("Measured ΔT (mK)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # (c) Recovered stress sum
    ax = axes[1, 0]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, recon_stress_sum / 1e6, cmap="RdBu_r", shading="auto")
    ax.set_title(f"Recovered σ₁+σ₂  (PSNR={metrics['PSNR']:.1f} dB)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # (d) Error
    ax = axes[1, 1]
    err = (gt_stress_sum - recon_stress_sum) / 1e6
    im = ax.pcolormesh(X * 1e3, Y * 1e3, err, cmap="bwr", shading="auto")
    ax.set_title(f"Error  (RMSE={metrics['RMSE_MPa']:.2f} MPa, CC={metrics['CC']:.4f})")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    plt.suptitle("Thermoelastic Stress Analysis — Plate with Hole", fontsize=14, y=1.02)
    plt.tight_layout()
    
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics
