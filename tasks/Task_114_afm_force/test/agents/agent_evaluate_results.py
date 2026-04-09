import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(z, F_gt, F_recon, delta_f, results_dir, assets_dir):
    """
    Compute metrics (PSNR, CC, RMSE), save results, and generate visualizations.
    
    Parameters:
    -----------
    z : ndarray
        Distance grid (m)
    F_gt : ndarray
        Ground truth force (N)
    F_recon : ndarray
        Reconstructed force (N)
    delta_f : ndarray
        Frequency shift (Hz)
    results_dir : str
        Path to results directory
    assets_dir : str
        Path to assets directory
        
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, CC, RMSE, and scale_factor
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Compute metrics
    # Use only the region where force is significant
    F_max = np.max(np.abs(F_gt))
    mask = np.abs(F_gt) > 0.01 * F_max
    if np.sum(mask) < 10:
        mask = np.ones(len(z), dtype=bool)

    gt = F_gt[mask]
    rc = F_recon[mask]

    # Scale the reconstruction to match GT (least-squares scaling)
    scale = np.sum(gt * rc) / (np.sum(rc * rc) + 1e-30)
    rc_scaled = rc * scale

    # RMSE
    rmse = np.sqrt(np.mean((gt - rc_scaled)**2))

    # PSNR (relative to signal range)
    signal_range = np.max(gt) - np.min(gt)
    mse = np.mean((gt - rc_scaled)**2)
    psnr = 10 * np.log10(signal_range**2 / (mse + 1e-30))

    # CC (scale-invariant)
    g = gt - np.mean(gt)
    r = rc - np.mean(rc)
    cc = np.sum(g * r) / (np.sqrt(np.sum(g**2) * np.sum(r**2)) + 1e-12)

    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RMSE": float(rmse),
        "scale_factor": float(scale),
    }
    
    # Apply scale factor for visualization and saving
    F_recon_scaled = F_recon * scale
    
    # Visualization
    z_nm = z * 1e9  # convert to nm for plotting
    F_gt_nN = F_gt * 1e9  # convert to nN
    F_recon_nN = F_recon_scaled * 1e9
    delta_f_Hz = delta_f  # already in Hz

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: GT force curve
    axes[0, 0].plot(z_nm, F_gt_nN, "b-", linewidth=2, label="GT Force F(z)")
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Distance z (nm)", fontsize=12)
    axes[0, 0].set_ylabel("Force (nN)", fontsize=12)
    axes[0, 0].set_title("Ground Truth: Lennard-Jones Force", fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_xlim([0, 5])

    # Panel 2: Frequency shift (observable)
    axes[0, 1].plot(z_nm, delta_f_Hz, "g-", linewidth=1.5, label="Δf(d)")
    axes[0, 1].set_xlabel("Distance d (nm)", fontsize=12)
    axes[0, 1].set_ylabel("Frequency shift Δf (Hz)", fontsize=12)
    axes[0, 1].set_title("FM-AFM Observable: Frequency Shift", fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_xlim([0, 5])

    # Panel 3: GT vs Reconstructed force
    axes[1, 0].plot(z_nm, F_gt_nN, "b-", linewidth=2, label="GT Force")
    axes[1, 0].plot(z_nm, F_recon_nN, "r--", linewidth=2, label="Sader-Jarvis Recon")
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].set_xlabel("Distance z (nm)", fontsize=12)
    axes[1, 0].set_ylabel("Force (nN)", fontsize=12)
    axes[1, 0].set_title(
        f"Force Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"CC={metrics['CC']:.4f}",
        fontsize=12,
    )
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].set_xlim([0, 5])

    # Panel 4: Error
    error_nN = np.abs(F_gt_nN - F_recon_nN)
    axes[1, 1].semilogy(z_nm, error_nN + 1e-15, "m-", linewidth=1.5)
    axes[1, 1].set_xlabel("Distance z (nm)", fontsize=12)
    axes[1, 1].set_ylabel("|Error| (nN)", fontsize=12)
    axes[1, 1].set_title(f"Absolute Error (RMSE={metrics['RMSE']:.2e} N)", fontsize=12)
    axes[1, 1].set_xlim([0, 5])

    plt.tight_layout()
    for p in [os.path.join(results_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save data files
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), F_gt)
        np.save(os.path.join(d, "recon_output.npy"), F_recon_scaled)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics
