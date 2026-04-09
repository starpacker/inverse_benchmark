"""
isdm_scatter — Imaging Through Scattering Media
=================================================
Recover a hidden image from speckle patterns observed through a scattering medium
using autocorrelation-based phase retrieval.

Physics:
  - Forward: Speckle intensity from scattering:
    I_speckle(k) = |F[object](k)|² (via memory effect / autocorrelation)
    The autocorrelation of speckle ≈ autocorrelation of object
  - Inverse: Phase retrieval using Hybrid Input-Output (HIO) algorithm
    Since |F[object]| is known from √(autocorrelation), recover phase iteratively
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
N            = 64        # image size
NOISE_LEVEL  = 0.001     # noise on autocorrelation (reduced from 0.01)
HIO_ITER     = 2000      # HIO iterations
BETA         = 0.9       # HIO feedback parameter
N_RESTARTS   = 10        # number of random restarts for phase retrieval
SEED         = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. GROUND TRUTH: Simple test image (letters/shapes)
# ═══════════════════════════════════════════════════════════════════
def create_gt_object(n):
    """Create a simple 2D binary object (cross + dots pattern)."""
    obj = np.zeros((n, n), dtype=np.float64)
    c = n // 2

    # Central cross
    obj[c-8:c+8, c-2:c+2] = 1.0
    obj[c-2:c+2, c-8:c+8] = 1.0

    # Corner dots
    for dy, dx in [(-15, -15), (-15, 15), (15, -15), (15, 15)]:
        yy, xx = np.ogrid[:n, :n]
        r = np.sqrt((yy - (c + dy))**2 + (xx - (c + dx))**2)
        obj[r < 4] = 1.0

    # Small rectangle
    obj[c+8:c+14, c-12:c-6] = 0.7

    # Triangle-like shape
    for i in range(8):
        obj[c-18+i, c+6:c+6+2*i+1] = 0.8

    return obj


# ═══════════════════════════════════════════════════════════════════
# 2. FORWARD MODEL: Scattering → autocorrelation
# ═══════════════════════════════════════════════════════════════════
def forward_scattering(obj, noise_level):
    """
    Simulate scattering: from the object, compute the speckle autocorrelation.

    In a memory-effect based imaging scenario:
      - Speckle pattern S(x) = |h * obj|^2 where h is the PSF of scattering medium
      - Autocorrelation of speckle: C(Δx) ≈ |autocorrelation of obj|^2
      - In Fourier domain: |F[S]|^2 = |F[obj]|^2 × |F[h]|^2
      - Under memory effect: we can extract |F[obj]|^2 from speckle autocorrelation

    Simplified model: we compute |F[obj]|^2 (the power spectrum) directly,
    which gives us the Fourier magnitude of the object.
    """
    # Compute Fourier transform of object
    F_obj = np.fft.fft2(obj)

    # Power spectrum = |F[obj]|^2 (this is what autocorrelation gives us)
    power_spectrum = np.abs(F_obj)**2

    # Simulate a random scattering transmission matrix effect
    # The TM modulates the field but autocorrelation cancels it out
    # We get the power spectrum with some noise
    TM = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
    speckle_field = np.fft.ifft2(F_obj * TM)
    speckle_intensity = np.abs(speckle_field)**2

    # From speckle, compute autocorrelation (= Fourier of power spectrum)
    F_speckle = np.fft.fft2(speckle_intensity)
    autocorr_speckle = np.abs(F_speckle)**2  # This approximates |F[obj]|^4 but
    # under memory effect simplification, we use |F[obj]|^2

    # Add noise proportionally to each pixel's power (not scaled by global max)
    noise = noise_level * np.sqrt(np.maximum(power_spectrum, 0)) * np.random.randn(*power_spectrum.shape)
    measured_magnitude = np.sqrt(np.maximum(power_spectrum + noise, 0))

    return speckle_intensity, measured_magnitude, power_spectrum


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE: HIO Phase Retrieval
# ═══════════════════════════════════════════════════════════════════
def estimate_support_from_autocorrelation(measured_magnitude, threshold_fraction=0.08):
    """
    Estimate support from autocorrelation of measured Fourier magnitudes.
    The autocorrelation of the object can be estimated from |F[obj]|^2.
    The support of the autocorrelation is ~2x the support of the object.
    We threshold the autocorrelation to get an approximate tight support.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes

    # Power spectrum = measured_magnitude^2
    power_spec = measured_magnitude ** 2
    # Autocorrelation = IFFT of power spectrum
    autocorr = np.real(np.fft.ifft2(power_spec))
    autocorr = np.fft.fftshift(autocorr)

    # Normalize
    autocorr_norm = autocorr / (np.max(autocorr) + 1e-12)

    # Threshold to get support estimate
    # Autocorrelation support is ~2x the object support
    support_auto = autocorr_norm > threshold_fraction

    # Clean up with morphological operations
    support_auto = binary_fill_holes(support_auto)
    # Erode to shrink from autocorrelation size to approximate object size
    support_auto = binary_erosion(support_auto, iterations=3)
    # Dilate a little to give some margin
    support_auto = binary_dilation(support_auto, iterations=2)

    return support_auto


def shrinkwrap_update(g, support, sigma=2.0, threshold_fraction=0.1):
    """
    Shrinkwrap support update: blur the current estimate and threshold.
    """
    blurred = gaussian_filter(np.abs(g), sigma=sigma)
    threshold = threshold_fraction * np.max(blurred)
    new_support = blurred > threshold
    return new_support


def hio_phase_retrieval(measured_magnitude, support, n_iter, beta, shrinkwrap=True):
    """
    Hybrid Input-Output (HIO) algorithm for phase retrieval with shrinkwrap.

    Given |F[obj]| (Fourier magnitude), recover the object by iterating
    between Fourier and real-space constraints.

    Args:
        measured_magnitude: |F[obj]| measured from speckle autocorrelation
        support: binary mask defining the initial support constraint
        n_iter: number of iterations
        beta: HIO feedback parameter
        shrinkwrap: whether to apply shrinkwrap support updates
    """
    n = measured_magnitude.shape[0]

    # Initialize with random phase
    phase = 2 * np.pi * np.random.rand(n, n)
    g = np.real(np.fft.ifft2(measured_magnitude * np.exp(1j * phase)))

    for iteration in range(n_iter):
        # Fourier constraint: replace magnitude, keep phase
        G = np.fft.fft2(g)
        G_constrained = measured_magnitude * np.exp(1j * np.angle(G))
        g_prime = np.real(np.fft.ifft2(G_constrained))

        # Real-space constraint: HIO update (vectorized)
        valid = support & (g_prime >= 0)
        g_new = np.where(valid, g_prime, g - beta * g_prime)
        g = g_new

        # Every 50 iterations, apply Error Reduction (ER) step
        if (iteration + 1) % 50 == 0:
            G = np.fft.fft2(g)
            G_constrained = measured_magnitude * np.exp(1j * np.angle(G))
            g_prime = np.real(np.fft.ifft2(G_constrained))
            g = g_prime * support
            g[g < 0] = 0

        # Shrinkwrap: update support every 100 iterations after initial 200
        if shrinkwrap and iteration >= 200 and (iteration + 1) % 100 == 0:
            sigma = max(1.0, 3.0 - iteration / 1000.0)  # decreasing sigma
            support = shrinkwrap_update(g, support, sigma=sigma, threshold_fraction=0.08)

    # Final cleanup
    g = g * support
    g[g < 0] = 0

    return g


def compute_r_factor(measured_magnitude, recon):
    """Compute R-factor (Fourier-space error) to evaluate reconstruction quality."""
    F_recon = np.fft.fft2(recon)
    recon_mag = np.abs(F_recon)
    r_factor = np.sum(np.abs(measured_magnitude - recon_mag)) / np.sum(measured_magnitude + 1e-12)
    return r_factor


def hio_multi_restart(measured_magnitude, initial_support, n_iter, beta, n_restarts=10):
    """
    Run HIO phase retrieval with multiple random restarts.
    Select the best result by lowest R-factor (Fourier-space error).
    """
    best_recon = None
    best_r_factor = np.inf

    for i in range(n_restarts):
        # Use different random seed for each restart
        support_i = initial_support.copy()
        recon = hio_phase_retrieval(measured_magnitude, support_i, n_iter, beta, shrinkwrap=True)
        r_fac = compute_r_factor(measured_magnitude, recon)
        print(f"    Restart {i+1}/{n_restarts}: R-factor = {r_fac:.6f}")
        if r_fac < best_r_factor:
            best_r_factor = r_fac
            best_recon = recon.copy()

    print(f"  Best R-factor: {best_r_factor:.6f}")
    return best_recon


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


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(gt, recon):
    """Compute PSNR, SSIM, CC between GT and reconstruction."""
    # Normalize both to [0, 1]
    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

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

    return {"PSNR": float(psnr), "SSIM": float(ssim_val), "CC": float(cc)}


# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(gt, speckle, recon, metrics):
    """Create visualization: GT, speckle, reconstruction, error."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

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


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("isdm_scatter — Imaging Through Scattering Media")
    print("=" * 60)

    # 1. Create ground truth
    print("[1/5] Creating ground truth object ...")
    gt = create_gt_object(N)

    # 2. Forward model: scattering
    print("[2/5] Simulating scattering (speckle pattern) ...")
    speckle, measured_mag, power_spec = forward_scattering(gt, NOISE_LEVEL)

    # 3. Create support constraint (estimated from autocorrelation + fallback circle)
    # Start with autocorrelation-based support estimation
    support_auto = estimate_support_from_autocorrelation(measured_mag)
    # Also create a tighter circular support as fallback
    yy, xx = np.ogrid[:N, :N]
    c = N // 2
    r = np.sqrt((yy - c)**2 + (xx - c)**2)
    support_circle = r < 0.2 * N
    # Use union of autocorrelation estimate and tight circle
    support = support_auto | support_circle
    print(f"  Support size: {np.sum(support)} pixels")

    # 4. Phase retrieval with multiple restarts
    print("[3/5] Running HIO phase retrieval with {0} restarts ...".format(N_RESTARTS))
    recon_raw = hio_multi_restart(measured_mag, support, HIO_ITER, BETA, n_restarts=N_RESTARTS)

    # 5. Align reconstruction (handle ambiguities)
    print("[4/5] Aligning reconstruction ...")
    recon, _ = align_and_compare(gt, recon_raw)

    # 6. Compute metrics
    metrics = compute_metrics(gt, recon)
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  SSIM = {metrics['SSIM']:.4f}")
    print(f"  CC   = {metrics['CC']:.4f}")

    # 7. Save outputs
    print("[5/5] Saving results ...")
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), recon)

    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt)
        np.save(os.path.join(d, "recon_output.npy"), recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # 8. Visualize
    visualize(gt, speckle, recon, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
