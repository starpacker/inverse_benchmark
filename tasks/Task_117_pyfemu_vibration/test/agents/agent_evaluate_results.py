import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

def mac_value(phi_a, phi_b):
    """MAC between two mode shape vectors."""
    num = np.dot(phi_a, phi_b) ** 2
    den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b)
    return num / (den + 1e-30)

def evaluate_results(d_gt, d_recon, freqs_gt, freqs_recon, modes_gt, modes_recon,
                     freqs_obs, n_modes, n_elem, L_total, results_dir, assets_dir):
    """
    Evaluate reconstruction quality, compute metrics, save results, and visualize.
    
    Args:
        d_gt: Ground truth damage vector
        d_recon: Reconstructed damage vector
        freqs_gt: Ground truth frequencies
        freqs_recon: Reconstructed frequencies
        modes_gt: Ground truth mode shapes
        modes_recon: Reconstructed mode shapes
        freqs_obs: Observed (noisy) frequencies
        n_modes: Number of modes
        n_elem: Number of elements
        L_total: Total beam length
        results_dir: Directory for saving results
        assets_dir: Directory for saving assets
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Compute metrics
    # PSNR on damage vector
    mse = np.mean((d_gt - d_recon) ** 2)
    data_range = max(np.max(d_gt) - np.min(d_gt), 0.01)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))
    
    # Correlation coefficient
    if np.std(d_gt) > 1e-10:
        cc = float(np.corrcoef(d_gt, d_recon)[0, 1])
    else:
        cc = 0.0
    
    rmse = float(np.sqrt(mse))
    
    # Frequency RMSE
    freq_rmse = float(np.sqrt(np.mean((freqs_gt - freqs_recon) ** 2)))
    
    # Average MAC
    mac_vals = []
    modes_recon_copy = modes_recon.copy()
    for j in range(n_modes):
        if np.dot(modes_gt[:, j], modes_recon_copy[:, j]) < 0:
            modes_recon_copy[:, j] *= -1
        mac_vals.append(mac_value(modes_gt[:, j], modes_recon_copy[:, j]))
    avg_mac = float(np.mean(mac_vals))
    
    # Damage localisation accuracy
    gt_damaged = set(np.where(d_gt > 0.05)[0])
    recon_damaged = set(np.where(d_recon > 0.05)[0])
    if len(gt_damaged) > 0:
        detection_rate = len(gt_damaged & recon_damaged) / len(gt_damaged) * 100
    else:
        detection_rate = 100.0
    
    metrics = {
        "PSNR": float(psnr),
        "CC": cc,
        "RMSE": rmse,
        "freq_RMSE_Hz": freq_rmse,
        "avg_MAC": avg_mac,
        "damage_detection_pct": detection_rate
    }
    
    # Print metrics
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.6f}")
    print(f"  Freq RMSE = {metrics['freq_RMSE_Hz']:.4f} Hz")
    print(f"  Avg MAC   = {metrics['avg_MAC']:.6f}")
    print(f"  Detection = {metrics['damage_detection_pct']:.0f}%")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), d_gt)
        np.save(os.path.join(d, "recon_output.npy"), d_recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    elem_centers = np.arange(n_elem) + 0.5
    
    # (a) Damage distribution
    ax = axes[0, 0]
    ax.bar(elem_centers - 0.2, d_gt, 0.4, label="True Damage", color="steelblue", alpha=0.8)
    ax.bar(elem_centers + 0.2, d_recon, 0.4, label="Identified Damage", color="salmon", alpha=0.8)
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Damage Parameter d")
    ax.set_title(f"Damage Identification  (PSNR={metrics['PSNR']:.1f} dB, CC={metrics['CC']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_elem)
    
    # (b) Frequency comparison
    ax = axes[0, 1]
    mode_idx = np.arange(1, n_modes + 1)
    ax.plot(mode_idx, freqs_gt, "bo-", label="GT Frequencies")
    ax.plot(mode_idx, freqs_obs, "g^--", label="Observed (noisy)", alpha=0.7)
    ax.plot(mode_idx, freqs_recon, "rs--", label="Identified Model")
    ax.set_xlabel("Mode Number")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Modal Frequencies  (freq RMSE={metrics['freq_RMSE_Hz']:.2f} Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Mode shapes (first 3)
    ax = axes[1, 0]
    n_dof = modes_gt.shape[0]
    x_nodes = np.linspace(0, L_total, n_dof)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for j in range(min(3, n_modes)):
        mg = modes_gt[:, j]
        mr = modes_recon_copy[:, j]
        ax.plot(x_nodes, mg, "-", color=colors[j], lw=2, label=f"Mode {j+1} GT")
        ax.plot(x_nodes, mr, "--", color=colors[j], lw=2, label=f"Mode {j+1} Identified")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Mode Shape Amplitude")
    ax.set_title(f"Mode Shapes  (avg MAC={metrics['avg_MAC']:.4f})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # (d) Residual / damage error
    ax = axes[1, 1]
    residual = d_gt - d_recon
    ax.bar(elem_centers, residual, 0.6, color="purple", alpha=0.6)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Damage Error (GT − Identified)")
    ax.set_title(f"Damage Residual  (RMSE={metrics['RMSE']:.4f}, Detection={metrics['damage_detection_pct']:.0f}%)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_elem)
    
    plt.suptitle("Vibration-Based Damage Identification — FE Model Updating", fontsize=14, y=1.01)
    plt.tight_layout()
    
    for p in [os.path.join(results_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics
