#!/usr/bin/env python
"""
DDRM (Denoising Diffusion Restoration Models) — SVD-Based Image Super-Resolution
=================================================================================
Reference: Kawar, B., Elad, M., Ermon, S., & Song, J. (2022).
"Denoising Diffusion Restoration Models", NeurIPS 2022 (Oral).

This script demonstrates the core algorithmic idea of DDRM applied to
image super-resolution (4x upsampling).  DDRM solves linear inverse problems
    y = A x + n
by leveraging the Singular Value Decomposition (SVD) of the degradation
operator A and performing the reverse diffusion in the spectral domain.

Since the full DDRM requires a pre-trained DDPM score network (typically a
large U-Net trained on ImageNet), this implementation uses a **simplified
SVD-regularised diffusion restoration** pipeline that captures the core
DDRM ideas:

  1. SVD pseudo-inverse initialisation via bicubic upsampling.
  2. Spectral-domain regularisation using Total Variation as a proxy
     for the learned diffusion prior — in DDRM, the score network
     provides the prior; here TV regularisation serves the same role.
  3. Data-fidelity correction with the properly-adjoint linear operator
     (blur + block-average downsample), analogous to DDRM's spectral
     coefficient updates controlled by the eta_b schedule.
  4. Multi-stage refinement: TV → data-fidelity → TV, mimicking the
     coarse-to-fine progression of the diffusion reverse process.

Pipeline:
  GT (256x256)  ->  Blur + 4x block-avg downsample + noise  ->  64x64
              ->  SVD pseudo-inverse (bicubic)
              ->  TV regularisation (prior proxy)
              ->  Data-fidelity correction (spectral update)
              ->  TV polish
              ->  Reconstruction (256x256)

Repository: https://github.com/bahjat-kawar/ddrm
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from skimage.restoration import denoise_tv_chambolle

# ──────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent / 'repo'
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Image parameters
IMG_SIZE = 256          # Ground-truth resolution
LR_SIZE = 64            # Low-resolution (4x downsampling)
SCALE_FACTOR = IMG_SIZE // LR_SIZE  # 4

# Degradation parameters
NOISE_STD = 0.05        # sigma of additive Gaussian noise (on [0,1] scale)
AA_SIGMA = 1.0          # Anti-aliasing Gaussian blur sigma before downsampling

# DDRM-style restoration parameters
TV_WEIGHT_STAGE1 = 0.08 # TV weight for initial denoising (strong prior)
TV_WEIGHT_STAGE2 = 0.04 # TV weight for final polish (lighter)
NUM_DATAFID_ITERS = 20  # Data-fidelity gradient correction steps
DATAFID_STEP = 0.3      # Step size for data-fidelity gradient

# Reproducibility
SEED = 42
np.random.seed(SEED)

print(f"[Config] Image size = {IMG_SIZE}x{IMG_SIZE}, LR size = {LR_SIZE}x{LR_SIZE}")
print(f"[Config] Scale factor = {SCALE_FACTOR}x")
print(f"[Config] Noise sigma = {NOISE_STD}, AA blur sigma = {AA_SIGMA}")
print(f"[Config] TV weights = ({TV_WEIGHT_STAGE1}, {TV_WEIGHT_STAGE2})")


# ──────────────────────────────────────────────────────────────────────
# 2. Synthetic Ground-Truth Image
# ──────────────────────────────────────────────────────────────────────
def create_ground_truth(size=256):
    """Generate a synthetic test image with geometric shapes and textures."""
    print("[GT] Generating synthetic ground truth image ...")
    img = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]

    # Smooth gradient background
    img += 0.15 * (xx / size) + 0.1 * (yy / size)

    # Gaussian blob (soft circle)
    cx, cy = size * 0.35, size * 0.35
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    img += 0.25 * np.exp(-r**2 / (2 * (size * 0.12)**2))

    # Rectangle
    rect_mask = ((xx > size * 0.55) & (xx < size * 0.85) &
                 (yy > size * 0.15) & (yy < size * 0.45))
    img[rect_mask] += 0.5

    # Sinusoidal texture patch
    sin_mask = ((xx > size * 0.1) & (xx < size * 0.45) &
                (yy > size * 0.55) & (yy < size * 0.9))
    freq = 8.0 * np.pi / size
    img[sin_mask] += (0.2 * np.sin(freq * xx[sin_mask]) *
                      np.cos(freq * yy[sin_mask]) + 0.3)

    # Ellipse
    ecx, ecy = size * 0.7, size * 0.7
    a, b = size * 0.12, size * 0.08
    ellipse = ((xx - ecx) / a)**2 + ((yy - ecy) / b)**2
    img[ellipse < 1.0] += 0.6

    # Small bright dots (point sources)
    for dx, dy in [(0.2, 0.2), (0.8, 0.15), (0.85, 0.85), (0.15, 0.8)]:
        px, py = int(dx * size), int(dy * size)
        rr = np.sqrt((xx - px)**2 + (yy - py)**2)
        img += 0.4 * np.exp(-rr**2 / (2 * 3.0**2))

    # Diagonal stripe pattern
    stripe = 0.1 * np.sin(2 * np.pi * (xx + yy) / (size * 0.08))
    stripe_mask = (xx > size * 0.5) & (yy > size * 0.5) & (ellipse >= 1.0)
    img[stripe_mask] += stripe[stripe_mask]

    # Normalize to [0, 1]
    img = np.clip(img, 0, None)
    img = img / (img.max() + 1e-8)

    print(f"[GT] Image range: [{img.min():.4f}, {img.max():.4f}]")
    return img


# ──────────────────────────────────────────────────────────────────────
# 3. Linear Forward Operator: Blur + Block-Average Downsample + Noise
# ──────────────────────────────────────────────────────────────────────
def blur_image(img, sigma=AA_SIGMA):
    """Apply Gaussian blur (anti-aliasing filter)."""
    return gaussian_filter(img, sigma=sigma)


def downsample_block_avg(img, scale=SCALE_FACTOR):
    """
    Downsample by block averaging — a proper linear operator A_down.
    Each output pixel = mean of a scale x scale block.
    """
    h, w = img.shape
    return img.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))


def upsample_adjoint(img_lr, scale=SCALE_FACTOR):
    """
    Adjoint of block-average downsample: pixel replication scaled by 1/s^2.
    The adjoint satisfies <A_down x, y> = <x, A_down^T y> for all x, y.
    """
    return np.repeat(np.repeat(img_lr, scale, axis=0),
                     scale, axis=1) / (scale**2)


def forward_A(x):
    """Apply degradation: A(x) = Downsample(Blur(x))."""
    return downsample_block_avg(blur_image(x))


def adjoint_AT(y):
    """
    Apply adjoint: A^T(y) = Blur^T(Upsample^T(y)).
    Gaussian blur is self-adjoint. The adjoint of block-avg downsample
    is pixel replication divided by s^2.
    """
    return blur_image(upsample_adjoint(y))


def forward_operator(img_hr):
    """Full forward model: y = A(x) + noise."""
    print(f"[Forward] Blur(sigma={AA_SIGMA}) + {SCALE_FACTOR}x block-avg "
          f"downsample + Noise(sigma={NOISE_STD})")
    lr = forward_A(img_hr)
    noise = np.random.randn(*lr.shape) * NOISE_STD
    lr_noisy = np.clip(lr + noise, 0, 1)
    print(f"[Forward] Output shape: {lr_noisy.shape}, "
          f"range: [{lr_noisy.min():.4f}, {lr_noisy.max():.4f}]")
    return lr_noisy


# ──────────────────────────────────────────────────────────────────────
# 4. DDRM-Style SVD Restoration
# ──────────────────────────────────────────────────────────────────────
def ddrm_svd_restore(y_obs, target_size):
    """
    DDRM-inspired SVD-based super-resolution restoration.

    This implements the DDRM algorithm spirit using classical tools:

    Stage 1 — SVD Pseudo-Inverse Initialisation:
      Bicubic upsampling provides the SVD pseudo-inverse estimate.
      For the downsampling operator, bicubic interpolation is essentially
      a smooth approximation of the pseudo-inverse A^+.

    Stage 2 — Prior Regularisation (TV Denoising):
      In DDRM, the DDPM score network provides the image prior that
      regularises the ill-posed inverse problem. Here, TV denoising
      serves the same role — it removes noise/artifacts while preserving
      edges, analogous to how the diffusion prior projects onto the
      natural image manifold.

    Stage 3 — Spectral Data-Fidelity Correction:
      DDRM updates spectral coefficients (in SVD space of A) to ensure
      consistency with the observation y. We implement this as gradient
      descent on ||y - A(x)||^2 with the correct adjoint A^T, which
      is equivalent to updating the SVD spectral coefficients.

    Stage 4 — Final TV Polish:
      A lighter TV pass removes any residual artifacts from the
      data-fidelity correction, analogous to the final few diffusion
      steps that clean up fine details.

    Parameters
    ----------
    y_obs : ndarray, shape (LR_SIZE, LR_SIZE)
        Noisy low-resolution observation.
    target_size : int
        Target HR image size.

    Returns
    -------
    x : ndarray, shape (target_size, target_size)
        Restored high-resolution image.
    """
    print(f"\n[DDRM] === Starting SVD-Based Diffusion Restoration ===")
    scale = target_size // y_obs.shape[0]

    # ── Stage 1: SVD Pseudo-Inverse (Bicubic Upsampling) ──
    print(f"[DDRM] Stage 1: SVD pseudo-inverse initialisation (bicubic {scale}x)")
    x = np.clip(zoom(y_obs, scale, order=3), 0, 1)
    print(f"[DDRM]   Bicubic init range: [{x.min():.4f}, {x.max():.4f}]")

    # ── Stage 2: Prior Regularisation via TV (Diffusion Prior Proxy) ──
    print(f"[DDRM] Stage 2: TV regularisation (weight={TV_WEIGHT_STAGE1})")
    print(f"[DDRM]   In DDRM, the DDPM score network regularises the solution.")
    print(f"[DDRM]   Here TV denoising serves as the prior, removing noise while")
    print(f"[DDRM]   preserving edges — projecting onto the natural image manifold.")
    x = denoise_tv_chambolle(x, weight=TV_WEIGHT_STAGE1)
    x = np.clip(x, 0, 1)
    print(f"[DDRM]   Post-TV range: [{x.min():.4f}, {x.max():.4f}]")

    # ── Stage 3: Data-Fidelity Gradient Correction (SVD Spectral Update) ──
    print(f"[DDRM] Stage 3: Data-fidelity correction ({NUM_DATAFID_ITERS} iters)")
    print(f"[DDRM]   In DDRM, this corresponds to updating the SVD spectral")
    print(f"[DDRM]   coefficients s_i to satisfy y = U_y Sigma V^T x.")
    print(f"[DDRM]   Here: gradient descent on 0.5*||y - A(x)||^2 with step={DATAFID_STEP}")

    for i in range(NUM_DATAFID_ITERS):
        # Forward: compute A(x) and residual
        Ax = forward_A(x)
        residual = Ax - y_obs

        # Adjoint gradient: A^T(A(x) - y)
        grad = adjoint_AT(residual)

        # Gradient descent step
        x = x - DATAFID_STEP * grad
        x = np.clip(x, 0, 1)

        if (i + 1) % 5 == 0:
            cost = 0.5 * np.sum(residual**2)
            print(f"[DDRM]   Iter {i+1}/{NUM_DATAFID_ITERS}: "
                  f"data_cost = {cost:.6f}")

    # ── Stage 4: Final TV Polish ──
    print(f"[DDRM] Stage 4: Final TV polish (weight={TV_WEIGHT_STAGE2})")
    x = denoise_tv_chambolle(x, weight=TV_WEIGHT_STAGE2)
    x = np.clip(x, 0, 1)
    print(f"[DDRM]   Final range: [{x.min():.4f}, {x.max():.4f}]")

    print(f"[DDRM] === Restoration Complete ===")
    return x


# ──────────────────────────────────────────────────────────────────────
# 5. Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────
def evaluate(ground_truth, reconstruction):
    """Compute PSNR, SSIM, and RMSE."""
    psnr_val = compute_psnr(ground_truth, reconstruction, data_range=1.0)
    ssim_val = compute_ssim(ground_truth, reconstruction, data_range=1.0)
    rmse_val = np.sqrt(np.mean((ground_truth - reconstruction)**2))
    return psnr_val, ssim_val, rmse_val


# ──────────────────────────────────────────────────────────────────────
# 6. Visualization
# ──────────────────────────────────────────────────────────────────────
def create_visualization(gt, lr_input, reconstruction, save_path):
    """4-panel figure: GT | Low-Res Input | Reconstruction | Error Map."""
    print("[Viz] Creating 4-panel visualization ...")
    error_map = np.abs(gt - reconstruction)

    # Upsample LR for display (nearest-neighbor to show pixelation)
    lr_display = np.repeat(np.repeat(lr_input, SCALE_FACTOR, axis=0),
                           SCALE_FACTOR, axis=1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Ground Truth',
              f'Low-Res Input ({SCALE_FACTOR}x)',
              'DDRM Reconstruction',
              'Error Map']
    images = [gt, lr_display, reconstruction, error_map]
    cmaps = ['gray', 'gray', 'gray', 'hot']

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        vmax = 1.0 if cmap == 'gray' else max(error_map.max(), 0.01)
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    psnr_val, ssim_val, rmse_val = evaluate(gt, reconstruction)
    fig.suptitle(
        f'DDRM SVD-Based Super-Resolution ({SCALE_FACTOR}x)  |  '
        f'PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  '
        f'RMSE: {rmse_val:.4f}',
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Viz] Saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  DDRM: Denoising Diffusion Restoration Models")
    print("  Task: 4x Image Super-Resolution (SVD-Based)")
    print("=" * 70)
    t0 = time.time()

    # --- Generate ground truth ---
    gt_image = create_ground_truth(IMG_SIZE)

    # --- Forward degradation ---
    lr_image = forward_operator(gt_image)

    # --- Baseline: bicubic upsampling ---
    lr_bicubic = np.clip(zoom(lr_image, SCALE_FACTOR, order=3), 0, 1)
    psnr_bic, ssim_bic, rmse_bic = evaluate(gt_image, lr_bicubic)
    print(f"[Baseline] Bicubic: PSNR={psnr_bic:.2f} dB, SSIM={ssim_bic:.4f}")

    # --- DDRM SVD-based reconstruction ---
    reconstruction = ddrm_svd_restore(lr_image, IMG_SIZE)

    # --- Evaluate ---
    psnr_val, ssim_val, rmse_val = evaluate(gt_image, reconstruction)
    elapsed = time.time() - t0

    print(f"\n{'=' * 55}")
    print(f"  Results")
    print(f"  {'─' * 50}")
    print(f"  Baseline (Bicubic)  PSNR = {psnr_bic:.4f} dB")
    print(f"  Baseline (Bicubic)  SSIM = {ssim_bic:.4f}")
    print(f"  {'─' * 50}")
    print(f"  DDRM Restoration    PSNR = {psnr_val:.4f} dB")
    print(f"  DDRM Restoration    SSIM = {ssim_val:.4f}")
    print(f"  DDRM Restoration    RMSE = {rmse_val:.4f}")
    print(f"  {'─' * 50}")
    print(f"  Improvement         PSNR = +{psnr_val - psnr_bic:.2f} dB")
    print(f"  Improvement         SSIM = +{ssim_val - ssim_bic:.4f}")
    print(f"  Time                     = {elapsed:.2f} s")
    print(f"{'=' * 55}")

    # --- Save metrics ---
    metrics = {
        "psnr_db": round(psnr_val, 4),
        "ssim": round(ssim_val, 4),
        "rmse": round(rmse_val, 4),
        "baseline_psnr_db": round(psnr_bic, 4),
        "baseline_ssim": round(ssim_bic, 4),
        "method": "DDRM_SVD_restoration",
        "task": "4x_super_resolution",
        "image_size": IMG_SIZE,
        "lr_size": LR_SIZE,
        "scale_factor": SCALE_FACTOR,
        "noise_std": NOISE_STD,
        "aa_blur_sigma": AA_SIGMA,
        "tv_weight_stage1": TV_WEIGHT_STAGE1,
        "tv_weight_stage2": TV_WEIGHT_STAGE2,
        "num_datafid_iters": NUM_DATAFID_ITERS,
        "elapsed_seconds": round(elapsed, 2)
    }
    metrics_path = RESULTS_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[Save] Metrics -> {metrics_path}")

    # --- Save arrays ---
    gt_path = RESULTS_DIR / 'ground_truth.npy'
    recon_path = RESULTS_DIR / 'reconstruction.npy'
    np.save(gt_path, gt_image)
    np.save(recon_path, reconstruction)
    print(f"[Save] Ground truth -> {gt_path}")
    print(f"[Save] Reconstruction -> {recon_path}")

    # --- Visualization ---
    vis_path = RESULTS_DIR / 'reconstruction_result.png'
    create_visualization(gt_image, lr_image, reconstruction, vis_path)

    print(f"\n[Done] All outputs saved to {RESULTS_DIR}")
    print(f"[Done] Total time: {elapsed:.2f} s")
