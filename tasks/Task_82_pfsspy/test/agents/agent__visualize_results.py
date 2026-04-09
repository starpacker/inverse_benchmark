import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

from scipy.ndimage import zoom

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def _visualize_results(br_clean, br_noisy, br_recon, br_ss, metrics, rss, save_path):
    """Generate comprehensive PFSS visualization (helper function)."""
    # Resize clean and noisy for comparison
    if br_clean.shape != br_recon.shape:
        zoom_factors = [br_recon.shape[i] / br_clean.shape[i] for i in range(2)]
        br_clean_r = zoom(br_clean, zoom_factors)
        br_noisy_r = zoom(br_noisy, zoom_factors)
    else:
        br_clean_r = br_clean
        br_noisy_r = br_noisy
    
    vmax = max(np.abs(br_clean_r).max(), np.abs(br_recon).max())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Clean magnetogram
    im0 = axes[0, 0].imshow(br_clean_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 0].set_title('GT Magnetogram (clean)')
    axes[0, 0].set_xlabel('Longitude (px)')
    axes[0, 0].set_ylabel('Sine Latitude (px)')
    plt.colorbar(im0, ax=axes[0, 0], label='B_r (G)')
    
    # (b) Noisy magnetogram (input)
    im1 = axes[0, 1].imshow(br_noisy_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 1].set_title('Input Magnetogram (noisy)')
    axes[0, 1].set_xlabel('Longitude (px)')
    plt.colorbar(im1, ax=axes[0, 1], label='B_r (G)')
    
    # (c) Reconstructed B_r at photosphere
    im2 = axes[0, 2].imshow(br_recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 2].set_title('PFSS Reconstructed B_r')
    axes[0, 2].set_xlabel('Longitude (px)')
    plt.colorbar(im2, ax=axes[0, 2], label='B_r (G)')
    
    # (d) Error map
    error = br_clean_r - br_recon
    im3 = axes[1, 0].imshow(error, cmap='seismic', 
                             vmin=-vmax*0.3, vmax=vmax*0.3,
                             aspect='auto', origin='lower')
    axes[1, 0].set_title('Error (GT - Recon)')
    axes[1, 0].set_xlabel('Longitude (px)')
    axes[1, 0].set_ylabel('Sine Latitude (px)')
    plt.colorbar(im3, ax=axes[1, 0], label='ΔB_r (G)')
    
    # (e) B_r at source surface
    im4 = axes[1, 1].imshow(br_ss, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1, 1].set_title(f'B_r at Source Surface (R={rss} R_sun)')
    axes[1, 1].set_xlabel('Longitude (px)')
    plt.colorbar(im4, ax=axes[1, 1], label='B_r (G)')
    
    # (f) Scatter: GT vs Recon
    axes[1, 2].scatter(br_clean_r.flatten(), br_recon.flatten(), 
                       alpha=0.3, s=5, c='steelblue')
    lim = vmax * 1.1
    axes[1, 2].plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Identity')
    axes[1, 2].set_xlabel('GT B_r (G)')
    axes[1, 2].set_ylabel('Recon B_r (G)')
    axes[1, 2].set_title(f'GT vs Recon (CC={metrics["cc"]:.4f})')
    axes[1, 2].legend()
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_xlim([-lim, lim])
    axes[1, 2].set_ylim([-lim, lim])
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"pfsspy — PFSS Coronal Magnetic Field Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC={metrics['cc']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} G | RE={metrics['relative_error']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")
