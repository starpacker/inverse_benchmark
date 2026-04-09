import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(gt_slip, rec_slip, obs_coords, d_obs, d_pred, patches,
                     fault_length, fault_width, nx_fault, ny_fault):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Computes metrics:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - CC: Correlation Coefficient
    - RMSE: Root Mean Square Error
    
    Generates plots showing:
    - Ground truth vs reconstructed slip distribution
    - Observed vs predicted surface displacements
    
    Args:
        gt_slip: ground truth slip distribution (ny, nx)
        rec_slip: reconstructed slip distribution (ny, nx)
        obs_coords: observation station coordinates (N_obs, 2)
        d_obs: observed displacement vector (3*N_obs,)
        d_pred: predicted displacement vector (3*N_obs,)
        patches: list of fault patch parameters
        fault_length: fault length in km
        fault_width: fault width in km
        nx_fault: number of patches along strike
        ny_fault: number of patches along dip
    
    Returns:
        dict containing computed metrics
    """
    print("[7] Computing metrics ...")

    gt_range = gt_slip.max() - gt_slip.min()
    if gt_range < 1e-15:
        gt_range = 1.0
    psnr = float(peak_signal_noise_ratio(gt_slip, rec_slip, data_range=gt_range))

    data_range = gt_range
    min_side = min(gt_slip.shape)
    win = min(7, min_side)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    ssim_val = float(ssim(gt_slip, rec_slip, data_range=data_range, win_size=win))

    gt_z = gt_slip - gt_slip.mean()
    rec_z = rec_slip - rec_slip.mean()
    denom = np.sqrt(np.sum(gt_z**2) * np.sum(rec_z**2))
    cc = float(np.sum(gt_z * rec_z) / denom) if denom > 1e-15 else 0.0

    rmse = float(np.sqrt(np.mean((gt_slip - rec_slip)**2)))

    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    SSIM = {ssim_val:.4f}")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.4f} m")

    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "CC": float(cc),
        "RMSE": float(rmse),
    }

    print("[8] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_slip)
        np.save(os.path.join(d, "recon_output.npy"), rec_slip)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("[9] Plotting ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.imshow(gt_slip, cmap='hot_r', origin='lower', aspect='auto',
                   extent=[0, fault_length, 0, fault_width])
    ax.set_title("Ground Truth Slip Distribution", fontsize=13)
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")
    plt.colorbar(im, ax=ax, label="Slip (m)")

    ax = axes[0, 1]
    im = ax.imshow(rec_slip, cmap='hot_r', origin='lower', aspect='auto',
                   extent=[0, fault_length, 0, fault_width])
    ax.set_title(f"Reconstructed Slip\nPSNR={metrics['PSNR']:.1f}dB, "
                 f"SSIM={metrics['SSIM']:.3f}, CC={metrics['CC']:.3f}", fontsize=12)
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")
    plt.colorbar(im, ax=ax, label="Slip (m)")

    ax = axes[1, 0]
    uz_obs = d_obs[2::3]
    uz_pred = d_pred[2::3]
    sc = ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=uz_obs,
                    cmap='RdBu_r', s=30, edgecolors='k', linewidths=0.3)
    ax.set_title("Observed Vertical Displacement", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    plt.colorbar(sc, ax=ax, label="Uz (m)")

    ax = axes[1, 1]
    sc = ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=uz_pred,
                    cmap='RdBu_r', s=30, edgecolors='k', linewidths=0.3,
                    vmin=uz_obs.min(), vmax=uz_obs.max())
    ax.set_title("Predicted Vertical Displacement", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    plt.colorbar(sc, ax=ax, label="Uz (m)")

    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return metrics
