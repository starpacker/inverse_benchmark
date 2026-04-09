import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

def evaluate_results(eps_gt, eps_recon, bscan_noisy, results_dir, assets_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes metrics (PSNR, SSIM, CC), saves numpy arrays and JSON metrics,
    and generates visualization plots.
    
    Parameters:
    -----------
    eps_gt : ndarray
        Ground truth permittivity model, shape (nz, nx)
    eps_recon : ndarray
        Reconstructed permittivity model, shape (nz, nx)
    bscan_noisy : ndarray
        Noisy B-scan data, shape (nz, nx)
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
    
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, SSIM, and CC values
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Compute PSNR
    mse = np.mean((eps_gt - eps_recon)**2)
    if mse < 1e-15:
        psnr_val = 100.0
    else:
        data_range = np.max(eps_gt) - np.min(eps_gt)
        psnr_val = 10 * np.log10(data_range**2 / mse)
    
    # Compute SSIM
    data_range_ssim = np.max(eps_gt) - np.min(eps_gt)
    if data_range_ssim < 1e-10:
        data_range_ssim = 1.0
    ssim_val = float(ssim(eps_gt, eps_recon, data_range=data_range_ssim))
    
    # Compute CC (Pearson correlation coefficient)
    g = eps_gt.ravel() - np.mean(eps_gt)
    r = eps_recon.ravel() - np.mean(eps_recon)
    denom = np.sqrt(np.sum(g**2) * np.sum(r**2))
    if denom < 1e-15:
        cc_val = 0.0
    else:
        cc_val = float(np.sum(g * r) / denom)
    
    metrics = {"PSNR": float(psnr_val), "SSIM": float(ssim_val), "CC": float(cc_val)}
    
    # Save metrics
    for path in [results_dir, assets_dir]:
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Save numpy outputs
    for path in [results_dir, assets_dir]:
        np.save(os.path.join(path, "gt_output.npy"), eps_gt)
        np.save(os.path.join(path, "recon_output.npy"), eps_recon)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # B-scan
    ax = axes[0, 0]
    im = ax.imshow(bscan_noisy, aspect='auto', cmap='seismic',
                   vmin=-np.max(np.abs(bscan_noisy)), vmax=np.max(np.abs(bscan_noisy)))
    ax.set_title("GPR B-scan (noisy)")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # GT permittivity
    ax = axes[0, 1]
    im = ax.imshow(eps_gt, aspect='auto', cmap='viridis')
    ax.set_title("GT Permittivity εᵣ")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="εᵣ")
    
    # Reconstructed permittivity
    ax = axes[1, 0]
    im = ax.imshow(eps_recon, aspect='auto', cmap='viridis',
                   vmin=eps_gt.min(), vmax=eps_gt.max())
    ax.set_title(f"Reconstructed εᵣ (PSNR={psnr_val:.1f}dB)")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="εᵣ")
    
    # Error map
    ax = axes[1, 1]
    error = np.abs(eps_gt - eps_recon)
    im = ax.imshow(error, aspect='auto', cmap='hot')
    ax.set_title(f"Absolute Error (SSIM={ssim_val:.3f}, CC={cc_val:.3f})")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="|error|")
    
    plt.suptitle("GPR Full-Waveform Inversion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for path in [results_dir, assets_dir]:
        fig.savefig(os.path.join(path, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(path, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics
