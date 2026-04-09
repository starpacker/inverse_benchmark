import matplotlib

matplotlib.use("Agg")

import json

import os

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import matplotlib.pyplot as plt

def evaluate_results(ground_truth, reconstruction, r_arr, z_arr, 
                     measurements, n_detectors, n_los_per_det, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes metrics (PSNR, SSIM, RMSE), generates visualization,
    and saves all artifacts to disk.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground truth emissivity field (NR, NZ)
    reconstruction : ndarray
        Reconstructed emissivity field (NR, NZ)
    r_arr : ndarray
        1D array of R coordinates
    z_arr : ndarray
        1D array of Z coordinates
    measurements : ndarray
        Noisy measurements (for sinogram visualization)
    n_detectors : int
        Number of detector fans
    n_los_per_det : int
        LOS per detector
    results_dir : str
        Directory to save results
        
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, RMSE values
    """
    print("[5/6] Computing metrics …")
    
    gt_2d = ground_truth
    recon_2d = reconstruction
    
    # Normalise reconstruction to same scale as GT for fair comparison
    if recon_2d.max() > 0:
        recon_2d = recon_2d / recon_2d.max() * gt_2d.max()
    
    # Compute metrics
    data_range = gt_2d.max() - gt_2d.min()
    if data_range == 0:
        data_range = 1.0
    
    psnr = peak_signal_noise_ratio(gt_2d, recon_2d, data_range=data_range)
    ssim = structural_similarity(gt_2d, recon_2d, data_range=data_range)
    rmse = np.sqrt(np.mean((gt_2d - recon_2d) ** 2))
    
    metrics = {
        "PSNR": round(float(psnr), 4),
        "SSIM": round(float(ssim), 4),
        "RMSE": round(float(rmse), 6)
    }
    
    print(f"       PSNR = {metrics['PSNR']:.2f} dB")
    print(f"       SSIM = {metrics['SSIM']:.4f}")
    print(f"       RMSE = {metrics['RMSE']:.6f}")
    
    # Save results
    print("[6/6] Saving results …")
    error_map = recon_2d - gt_2d
    
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_2d)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_2d)
    np.save(os.path.join(results_dir, "measurements.npy"), measurements)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"       Metrics → {os.path.join(results_dir, 'metrics.json')}")
    
    # Generate visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    _visualise(r_arr, z_arr, gt_2d, measurements, recon_2d, error_map, 
               metrics, n_detectors, n_los_per_det, vis_path)
    
    print("\n✓ Pipeline complete.")
    
    return metrics

def _visualise(r_arr, z_arr, gt, sinogram, recon, error, metrics, 
               n_detectors, n_los_per_det, save_path):
    """4-panel figure: GT, sinogram, reconstruction, error map."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent_rz = [r_arr[0], r_arr[-1], z_arr[0], z_arr[-1]]

    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(gt.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title("(a) Ground Truth Emissivity ε(R,Z)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (b) Line-integrated measurements (sinogram-like)
    ax = axes[0, 1]
    n_det = n_detectors
    n_los = n_los_per_det
    sino_2d = sinogram.reshape(n_det, n_los) if len(sinogram) == n_det * n_los \
        else sinogram.reshape(-1, n_los)[:n_det]
    im = ax.imshow(sino_2d, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("(b) Line-Integrated Measurements")
    ax.set_xlabel("LOS index")
    ax.set_ylabel("Detector fan")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (c) Reconstruction
    ax = axes[1, 0]
    im = ax.imshow(recon.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title(f"(c) Reconstruction  PSNR={metrics['PSNR']:.1f} dB  "
                 f"SSIM={metrics['SSIM']:.3f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (d) Error map
    ax = axes[1, 1]
    im = ax.imshow(error.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="seismic",
                   vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    ax.set_title(f"(d) Error Map  RMSE={metrics['RMSE']:.4f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Plasma/Fusion Tomography — Tokamak Emissivity Reconstruction",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualisation → {save_path}")
