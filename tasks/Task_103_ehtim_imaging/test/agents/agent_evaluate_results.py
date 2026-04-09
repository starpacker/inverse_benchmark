import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def compute_metrics(gt, rec):
    """Compute PSNR, SSIM, and cross-correlation metrics."""
    gt_n = gt / gt.max() if gt.max() > 0 else gt
    rec_n = rec / rec.max() if rec.max() > 0 else rec
    mse = np.mean((gt_n - rec_n) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
    dr = max(gt_n.max() - gt_n.min(), rec_n.max() - rec_n.min())
    if dr < 1e-15:
        dr = 1.0
    ssim_val = ssim(gt_n, rec_n, data_range=dr)
    gz = gt_n - gt_n.mean()
    rz = rec_n - rec_n.mean()
    d = np.sqrt(np.sum(gz ** 2) * np.sum(rz ** 2))
    cc = np.sum(gz * rz) / d if d > 1e-15 else 0.0
    return float(psnr), float(ssim_val), float(cc)

def evaluate_results(data, inversion_result, fov_uas):
    """
    Evaluate reconstruction results and generate outputs.
    
    Computes metrics, saves results, and generates visualization plots.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data
    inversion_result : dict
        Dictionary from run_inversion
    fov_uas : float
        Field of view in micro-arcseconds
        
    Returns
    -------
    dict
        Dictionary containing PSNR, SSIM, and CC metrics
    """
    gt_image = data['gt_image']
    u = data['u']
    v = data['v']
    cleaned = inversion_result['cleaned']
    dirty = inversion_result['dirty']
    
    # Compute metrics
    psnr, ssim_val, cc = compute_metrics(gt_image, cleaned)
    metrics = {"PSNR": float(psnr), "SSIM": float(ssim_val), "CC": float(cc)}
    
    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    SSIM = {ssim_val:.4f}")
    print(f"    CC   = {cc:.4f}")
    
    # Save outputs
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_image)
        np.save(os.path.join(d, "recon_output.npy"), cleaned)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ext = [-fov_uas / 2, fov_uas / 2, -fov_uas / 2, fov_uas / 2]
    
    ax = axes[0, 0]
    im = ax.imshow(gt_image, cmap='afmhot', origin='lower', extent=ext)
    ax.set_title("Ground Truth: Black Hole Shadow", fontsize=13)
    ax.set_xlabel("RA offset (μas)")
    ax.set_ylabel("Dec offset (μas)")
    plt.colorbar(im, ax=ax, label="Flux density")
    
    ax = axes[0, 1]
    ax.scatter(u, v, s=0.3, alpha=0.3, c='navy')
    ax.set_title(f"UV Coverage ({len(u)} points)", fontsize=13)
    ax.set_xlabel("u (cycles/μas)")
    ax.set_ylabel("v (cycles/μas)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    im = ax.imshow(dirty, cmap='afmhot', origin='lower', extent=ext)
    ax.set_title("Dirty Image", fontsize=13)
    ax.set_xlabel("RA offset (μas)")
    ax.set_ylabel("Dec offset (μas)")
    plt.colorbar(im, ax=ax, label="Flux density")
    
    ax = axes[1, 1]
    im = ax.imshow(cleaned, cmap='afmhot', origin='lower', extent=ext)
    ax.set_title(f"CLEAN Reconstruction\nPSNR={metrics['PSNR']:.1f}dB, "
                 f"SSIM={metrics['SSIM']:.3f}, CC={metrics['CC']:.3f}", fontsize=12)
    ax.set_xlabel("RA offset (μas)")
    ax.set_ylabel("Dec offset (μas)")
    plt.colorbar(im, ax=ax, label="Flux density")
    
    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics
