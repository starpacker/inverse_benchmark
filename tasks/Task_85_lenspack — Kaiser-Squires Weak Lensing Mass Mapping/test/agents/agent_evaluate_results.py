import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

import warnings

warnings.filterwarnings('ignore')

def evaluate_results(kappa_true, kE, kB, g1_true, g2_true, g1_obs, g2_obs, params, results_dir):
    """
    Evaluate mass mapping quality and save results.
    
    Args:
        kappa_true: True convergence map
        kE: Reconstructed E-mode convergence
        kB: Reconstructed B-mode convergence
        g1_true: True shear component 1
        g2_true: True shear component 2
        g1_obs: Observed shear component 1
        g2_obs: Observed shear component 2
        params: Dictionary of parameters
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from scipy.ndimage import uniform_filter, gaussian_filter
    
    # Remove mean (mass sheet degeneracy)
    gt = kappa_true - kappa_true.mean()
    recon = kE - kE.mean()
    
    # PSNR
    mse = np.mean((gt - recon)**2)
    data_range = gt.max() - gt.min()
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # SSIM
    def ssim_2d(img1, img2, win_size=7):
        C1 = (0.01 * data_range)**2
        C2 = (0.03 * data_range)**2
        mu1 = uniform_filter(img1, win_size)
        mu2 = uniform_filter(img2, win_size)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu12 = mu1 * mu2
        sigma1_sq = uniform_filter(img1**2, win_size) - mu1_sq
        sigma2_sq = uniform_filter(img2**2, win_size) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, win_size) - mu12
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    ssim = ssim_2d(gt, recon)
    
    # Correlation coefficient
    cc = np.corrcoef(gt.flatten(), recon.flatten())[0, 1]
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # Relative error
    re = rmse / np.sqrt(np.mean(gt**2)) if np.mean(gt**2) > 0 else float('inf')
    
    # Peak recovery
    gt_smooth = gaussian_filter(gt, sigma=3)
    recon_smooth = gaussian_filter(recon, sigma=3)
    gt_peak = np.unravel_index(np.argmax(gt_smooth), gt_smooth.shape)
    recon_peak = np.unravel_index(np.argmax(recon_smooth), recon_smooth.shape)
    peak_offset = np.sqrt((gt_peak[0] - recon_peak[0])**2 + (gt_peak[1] - recon_peak[1])**2)
    
    metrics = {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'cc': float(cc),
        'rmse': float(rmse),
        'relative_error': float(re),
        'peak_offset_pixels': float(peak_offset),
    }
    
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] SSIM = {metrics['ssim']:.6f}")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMSE = {metrics['rmse']:.6f}")
    print(f"[EVAL] Relative Error = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Peak offset = {metrics['peak_offset_pixels']:.2f} pixels")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    ny, nx = kappa_true.shape
    np.save(os.path.join(results_dir, "input.npy"), np.stack([g1_obs, g2_obs]))
    np.save(os.path.join(results_dir, "ground_truth.npy"), kappa_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), kE)
    print(f"[SAVE] Input shape: (2, {ny}, {nx}) → input.npy")
    print(f"[SAVE] GT shape: {kappa_true.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {kE.shape} → reconstruction.npy")
    
    # Visualization
    nx = params['nx']
    ny = params['ny']
    pixel_size = params['pixel_size']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    extent = [-nx / 2 * pixel_size, nx / 2 * pixel_size, -ny / 2 * pixel_size, ny / 2 * pixel_size]
    
    # (a) True convergence
    ax = axes[0, 0]
    im = ax.imshow(kappa_true, cmap='hot', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('True Convergence κ')
    plt.colorbar(im, ax=ax, label='κ')
    
    # (b) Observed shear field
    ax = axes[0, 1]
    gamma_mag = np.sqrt(g1_obs**2 + g2_obs**2)
    im2 = ax.imshow(gamma_mag, cmap='viridis', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('Observed |γ| (noisy)')
    plt.colorbar(im2, ax=ax, label='|γ|')
    
    # (c) KS93 reconstructed convergence (E-mode)
    ax = axes[0, 2]
    im3 = ax.imshow(kE, cmap='hot', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('KS93 Reconstructed κ_E')
    plt.colorbar(im3, ax=ax, label='κ_E')
    
    # (d) Error map
    ax = axes[1, 0]
    error = gt - recon
    vmax_err = np.max(np.abs(error)) * 0.8
    im4 = ax.imshow(error, cmap='seismic', origin='lower', extent=extent,
                    vmin=-vmax_err, vmax=vmax_err)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('Error (GT - Recon)')
    plt.colorbar(im4, ax=ax, label='Δκ')
    
    # (e) B-mode
    ax = axes[1, 1]
    im5 = ax.imshow(kB, cmap='RdBu_r', origin='lower', extent=extent)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title('B-mode κ_B (noise diagnostic)')
    plt.colorbar(im5, ax=ax, label='κ_B')
    
    # (f) Scatter: GT vs Recon
    ax = axes[1, 2]
    ax.scatter(gt.flatten(), recon.flatten(), s=1, alpha=0.3, c='steelblue')
    lim = max(np.abs(gt).max(), np.abs(recon).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Identity')
    ax.set_xlabel('True κ (zero-mean)')
    ax.set_ylabel('Recon κ (zero-mean)')
    ax.set_title(f'True vs Recon (CC={metrics["cc"]:.4f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"lenspack — Kaiser-Squires Weak Lensing Mass Mapping\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"CC={metrics['cc']:.4f} | RMSE={metrics['rmse']:.6f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics
