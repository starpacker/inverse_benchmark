import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def align_and_compare(gt, recon):
    """
    Phase retrieval has ambiguities (translation, inversion).
    Try all 4 flips and pick best correlation.
    """
    best_cc = -1
    best_recon = recon.copy()

    candidates = [
        recon,
        np.flipud(recon),
        np.fliplr(recon),
        np.flipud(np.fliplr(recon)),
    ]

    for cand in candidates:
        # Try all circular shifts to find best alignment
        F_gt = np.fft.fft2(gt)
        F_cand = np.fft.fft2(cand)
        cross_corr = np.real(np.fft.ifft2(F_gt * np.conj(F_cand)))
        shift = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)

        aligned = np.roll(np.roll(cand, shift[0], axis=0), shift[1], axis=1)

        # Compute CC
        gt_norm = gt - np.mean(gt)
        al_norm = aligned - np.mean(aligned)
        denom = np.sqrt(np.sum(gt_norm**2) * np.sum(al_norm**2))
        if denom > 0:
            cc = np.sum(gt_norm * al_norm) / denom
        else:
            cc = 0

        if cc > best_cc:
            best_cc = cc
            best_recon = aligned.copy()

    return best_recon, best_cc

def evaluate_results(gt, recon_raw, speckle):
    """
    Align reconstruction, compute metrics, save outputs, and create visualization.
    
    Args:
        gt: Ground truth object
        recon_raw: Raw reconstruction from phase retrieval
        speckle: Speckle pattern for visualization
    
    Returns:
        metrics: Dictionary containing PSNR, SSIM, CC
        recon_aligned: Aligned reconstruction
    """
    # Align reconstruction (handle ambiguities)
    recon_aligned, _ = align_and_compare(gt, recon_raw)
    
    # Normalize both to [0, 1]
    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon_aligned - recon_aligned.min()) / (recon_aligned.max() - recon_aligned.min() + 1e-12)

    # PSNR
    mse = np.mean((gt_n - re_n)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    # SSIM
    ssim_val = ssim(gt_n, re_n, data_range=1.0)

    # CC
    g = gt_n - np.mean(gt_n)
    r = re_n - np.mean(re_n)
    denom = np.sqrt(np.sum(g**2) * np.sum(r**2))
    cc = np.sum(g * r) / (denom + 1e-12)

    metrics = {"PSNR": float(psnr), "SSIM": float(ssim_val), "CC": float(cc)}
    
    # Save outputs
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), recon_aligned)

    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt)
        np.save(os.path.join(d, "recon_output.npy"), recon_aligned)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    im0 = axes[0, 0].imshow(gt_n, cmap="gray")
    axes[0, 0].set_title("Ground Truth Object", fontsize=14)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.log1p(speckle), cmap="hot")
    axes[0, 1].set_title("Speckle Pattern (log scale)", fontsize=14)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(re_n, cmap="gray")
    axes[1, 0].set_title(
        f"HIO Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"SSIM={metrics['SSIM']:.4f}, CC={metrics['CC']:.4f}",
        fontsize=12,
    )
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    error = np.abs(gt_n - re_n)
    im3 = axes[1, 1].imshow(error, cmap="magma")
    axes[1, 1].set_title(f"Absolute Error (RMSE={np.sqrt(np.mean(error**2)):.4f})", fontsize=12)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics, recon_aligned
