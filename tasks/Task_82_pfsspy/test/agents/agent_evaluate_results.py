import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from scipy.ndimage import zoom

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def evaluate_results(br_clean, br_noisy, output, br_recon, rss, results_dir):
    """
    Evaluate PFSS reconstruction quality and generate visualizations.
    
    Compare the reconstructed photospheric B_r (from PFSS solution)
    with the clean (noise-free) input magnetogram.
    Also evaluate field properties at different heights.
    
    Args:
        br_clean: Clean (noise-free) ground truth magnetogram
        br_noisy: Noisy input magnetogram
        output: PFSS output object
        br_recon: Reconstructed B_r at photosphere
        rss: Source surface radius
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Get magnetic field components from PFSS output
    bg_raw = output.bg  # (nphi+1, ns+1, nr+1, 3) - B field on cell boundaries
    bg = np.array(bg_raw.value if hasattr(bg_raw, 'value') else bg_raw)
    
    # Resize clean to match recon shape if needed
    if br_clean.shape != br_recon.shape:
        zoom_factors = [br_recon.shape[i] / br_clean.shape[i] for i in range(2)]
        br_clean_resized = zoom(br_clean, zoom_factors)
    else:
        br_clean_resized = br_clean
    
    # Flatten for comparison
    gt = br_clean_resized.flatten()
    recon = br_recon.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((gt - recon)**2))
    
    # Correlation coefficient
    cc = np.corrcoef(gt, recon)[0, 1]
    
    # Relative error
    re = np.sqrt(np.mean((gt - recon)**2)) / np.sqrt(np.mean(gt**2))
    
    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - recon)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # Magnetic energy (proxy for field quality)
    # Total unsigned flux at photosphere
    total_flux_gt = np.sum(np.abs(gt))
    total_flux_recon = np.sum(np.abs(recon))
    flux_ratio = total_flux_recon / total_flux_gt if total_flux_gt > 0 else 0
    
    # B_r at source surface (should be ~0 for open field)
    br_ss_raw = output.bc[0][:, :, -1]
    br_ss = np.array(br_ss_raw.value if hasattr(br_ss_raw, 'value') else br_ss_raw).T
    max_br_ss = np.max(np.abs(br_ss))
    
    # Open flux
    open_flux = np.sum(np.abs(br_ss))
    
    metrics = {
        'psnr': float(psnr),
        'rmse': float(rmse),
        'cc': float(cc),
        'relative_error': float(re),
        'flux_ratio': float(flux_ratio),
        'max_br_source_surface': float(max_br_ss),
        'open_flux': float(open_flux),
        'br_recon_shape': list(br_recon.shape),
        'bg_shape': list(bg.shape),
    }
    
    # Print metrics
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMSE = {metrics['rmse']:.4f} G")
    print(f"[EVAL] Relative Error = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Flux ratio = {metrics['flux_ratio']:.4f}")
    print(f"[EVAL] Max B_r at source surface = {metrics['max_br_source_surface']:.4f} G")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), br_clean)
    np.save(os.path.join(results_dir, "reconstruction.npy"), br_recon)
    np.save(os.path.join(results_dir, "input.npy"), br_noisy)
    print(f"[SAVE] GT shape: {br_clean.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {br_recon.shape} → reconstruction.npy")
    print(f"[SAVE] Input shape: {br_noisy.shape} → input.npy")
    
    # Generate visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    _visualize_results(br_clean, br_noisy, br_recon, br_ss, metrics, rss, vis_path)
    
    return metrics

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
