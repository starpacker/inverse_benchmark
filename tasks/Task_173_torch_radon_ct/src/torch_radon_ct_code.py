"""
torch_radon_ct: CT Reconstruction via Radon Transform Inversion

Inverse Problem: Given a CT sinogram (Radon transform of an object),
reconstruct the original 2D image.

Forward model:  y = A*x + noise   (A = Radon transform)
Inverse:        x_hat = A^{-1}(y) (via FBP or iterative methods)

Methods implemented:
  1. Filtered Back Projection (FBP) — baseline analytical method
  2. SIRT (Simultaneous Iterative Reconstruction Technique) — iterative algebraic
  3. SIRT initialized from FBP — iterative refinement of FBP

Reference repo: https://github.com/matteo-ronchetti/torch-radon
Uses scikit-image radon/iradon as forward/adjoint operators.
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 128          # phantom resolution
NUM_ANGLES = 180          # projection angles
SNR_DB = 30.0             # signal-to-noise ratio for noisy sinogram

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(signal, snr_db):
    """Add Gaussian noise to achieve a target SNR in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def compute_metrics(ground_truth, reconstruction):
    """Compute PSNR, SSIM, and RMSE between ground truth and reconstruction."""
    gt = ground_truth.copy()
    rec = reconstruction.copy()
    rec = np.clip(rec, 0, None)

    gt_max = gt.max()
    if gt_max > 0:
        gt_norm = gt / gt_max
        rec_norm = rec / gt_max
        rec_norm = np.clip(rec_norm, 0, 1)
    else:
        gt_norm = gt
        rec_norm = rec

    psnr = peak_signal_noise_ratio(gt_norm, rec_norm, data_range=1.0)
    ssim = structural_similarity(gt_norm, rec_norm, data_range=1.0)
    rmse = np.sqrt(mean_squared_error(gt_norm, rec_norm))
    return psnr, ssim, rmse


def get_circle_mask(size):
    """Create a circular mask for the reconstruction domain."""
    center = size // 2
    Y, X = np.ogrid[:size, :size]
    return ((X - center)**2 + (Y - center)**2) <= center**2


def estimate_operator_norm(theta, output_size, n_iter=15):
    """Estimate ||A^T A|| via power iteration for step size selection."""
    x = np.random.RandomState(0).randn(output_size, output_size)
    norm_est = 1.0
    for _ in range(n_iter):
        Ax = radon(x, theta=theta, circle=True)
        AtAx = iradon(Ax, theta=theta, output_size=output_size, filter_name=None, circle=True)
        norm_est = np.sqrt(np.sum(AtAx**2) / (np.sum(x**2) + 1e-10))
        x = AtAx / (np.linalg.norm(AtAx) + 1e-10) * np.linalg.norm(x)
    return norm_est


# ---------------------------------------------------------------------------
# Reconstruction methods
# ---------------------------------------------------------------------------

def reconstruct_fbp(sinogram, theta, output_size, filter_name='ramp'):
    """Filtered Back Projection reconstruction."""
    return iradon(sinogram, theta=theta, output_size=output_size,
                  filter_name=filter_name, circle=True)


def reconstruct_sirt(sinogram, theta, output_size, n_iter=200, init_fbp=False):
    """
    SIRT (Simultaneous Iterative Reconstruction Technique).
    
    Landweber iteration: x_{k+1} = x_k + step * A^T(y - A*x_k)
    with non-negativity constraint.
    """
    norm_est = estimate_operator_norm(theta, output_size)
    step = 0.9 / (norm_est + 1e-6)
    circle_mask = get_circle_mask(output_size)

    if init_fbp:
        x = iradon(sinogram, theta=theta, output_size=output_size,
                   filter_name='ramp', circle=True)
        x = np.maximum(x, 0) * circle_mask
    else:
        x = np.zeros((output_size, output_size))

    for i in range(n_iter):
        Ax = radon(x, theta=theta, circle=True)
        residual = sinogram - Ax
        gradient = iradon(residual, theta=theta, output_size=output_size,
                         filter_name=None, circle=True)
        x = x + step * gradient
        x = np.maximum(x, 0) * circle_mask

    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CT Reconstruction via Radon Transform Inversion")
    print("=" * 60)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Generate phantom (ground truth)
    # ------------------------------------------------------------------
    print("\n[1] Generating Shepp-Logan phantom...")
    phantom_full = shepp_logan_phantom()
    phantom = resize(phantom_full, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
    phantom = phantom.astype(np.float64)
    print(f"    Phantom shape: {phantom.shape}, range: [{phantom.min():.4f}, {phantom.max():.4f}]")

    # ------------------------------------------------------------------
    # 2. Forward model: Radon transform + noise
    # ------------------------------------------------------------------
    print("\n[2] Computing sinogram (Radon transform)...")
    theta = np.linspace(0, 179, NUM_ANGLES, endpoint=True)
    sinogram_clean = radon(phantom, theta=theta, circle=True)
    print(f"    Sinogram shape: {sinogram_clean.shape}")

    sinogram_noisy = add_gaussian_noise(sinogram_clean, SNR_DB)
    actual_snr = 10 * np.log10(np.mean(sinogram_clean ** 2) /
                                np.mean((sinogram_noisy - sinogram_clean) ** 2))
    print(f"    Added noise: target SNR={SNR_DB:.1f} dB, actual SNR={actual_snr:.1f} dB")

    # ------------------------------------------------------------------
    # 3. Reconstruction
    # ------------------------------------------------------------------
    results = {}
    recons = {}

    # --- FBP (Ram-Lak) ---
    print("\n[3a] FBP reconstruction (Ram-Lak)...")
    t0 = time.time()
    recon = reconstruct_fbp(sinogram_noisy, theta, IMAGE_SIZE, 'ramp')
    dt = time.time() - t0
    p, s, r = compute_metrics(phantom, recon)
    print(f"    PSNR={p:.2f} dB, SSIM={s:.4f}, RMSE={r:.4f}, time={dt:.2f}s")
    results['FBP'] = {'psnr': p, 'ssim': s, 'rmse': r, 'time': dt}
    recons['FBP'] = recon

    # --- FBP (Shepp-Logan filter) ---
    print("\n[3b] FBP reconstruction (Shepp-Logan filter)...")
    t0 = time.time()
    recon = reconstruct_fbp(sinogram_noisy, theta, IMAGE_SIZE, 'shepp-logan')
    dt = time.time() - t0
    p, s, r = compute_metrics(phantom, recon)
    print(f"    PSNR={p:.2f} dB, SSIM={s:.4f}, RMSE={r:.4f}, time={dt:.2f}s")
    results['FBP-SL'] = {'psnr': p, 'ssim': s, 'rmse': r, 'time': dt}
    recons['FBP-SL'] = recon

    # --- FBP (Cosine filter) ---
    print("\n[3c] FBP reconstruction (Cosine filter)...")
    t0 = time.time()
    recon = reconstruct_fbp(sinogram_noisy, theta, IMAGE_SIZE, 'cosine')
    dt = time.time() - t0
    p, s, r = compute_metrics(phantom, recon)
    print(f"    PSNR={p:.2f} dB, SSIM={s:.4f}, RMSE={r:.4f}, time={dt:.2f}s")
    results['FBP-Cosine'] = {'psnr': p, 'ssim': s, 'rmse': r, 'time': dt}
    recons['FBP-Cosine'] = recon

    # --- FBP (Hamming filter) ---
    print("\n[3d] FBP reconstruction (Hamming filter)...")
    t0 = time.time()
    recon = reconstruct_fbp(sinogram_noisy, theta, IMAGE_SIZE, 'hamming')
    dt = time.time() - t0
    p, s, r = compute_metrics(phantom, recon)
    print(f"    PSNR={p:.2f} dB, SSIM={s:.4f}, RMSE={r:.4f}, time={dt:.2f}s")
    results['FBP-Hamming'] = {'psnr': p, 'ssim': s, 'rmse': r, 'time': dt}
    recons['FBP-Hamming'] = recon

    # --- SIRT from zeros ---
    print(f"\n[3e] SIRT reconstruction (200 iterations, from zeros)...")
    t0 = time.time()
    recon = reconstruct_sirt(sinogram_noisy, theta, IMAGE_SIZE, n_iter=200, init_fbp=False)
    dt = time.time() - t0
    p, s, r = compute_metrics(phantom, recon)
    print(f"    PSNR={p:.2f} dB, SSIM={s:.4f}, RMSE={r:.4f}, time={dt:.2f}s")
    results['SIRT'] = {'psnr': p, 'ssim': s, 'rmse': r, 'time': dt}
    recons['SIRT'] = recon

    # --- SIRT from FBP ---
    print(f"\n[3f] SIRT-FBP reconstruction (50 iterations, from FBP)...")
    t0 = time.time()
    recon = reconstruct_sirt(sinogram_noisy, theta, IMAGE_SIZE, n_iter=50, init_fbp=True)
    dt = time.time() - t0
    p, s, r = compute_metrics(phantom, recon)
    print(f"    PSNR={p:.2f} dB, SSIM={s:.4f}, RMSE={r:.4f}, time={dt:.2f}s")
    results['SIRT-FBP'] = {'psnr': p, 'ssim': s, 'rmse': r, 'time': dt}
    recons['SIRT-FBP'] = recon

    # ------------------------------------------------------------------
    # 4. Select best method
    # ------------------------------------------------------------------
    best_method = max(results, key=lambda k: results[k]['psnr'])
    best_recon = recons[best_method]
    best_metrics = results[best_method]

    # Best iterative
    iter_keys = [k for k in results if 'SIRT' in k]
    best_iter = max(iter_keys, key=lambda k: results[k]['psnr'])
    best_iter_recon = recons[best_iter]
    best_iter_metrics = results[best_iter]

    # Best FBP
    fbp_keys = [k for k in results if k.startswith('FBP')]
    best_fbp = max(fbp_keys, key=lambda k: results[k]['psnr'])
    best_fbp_recon = recons[best_fbp]
    best_fbp_metrics = results[best_fbp]

    print(f"\n[4] Best overall: {best_method} (PSNR={best_metrics['psnr']:.2f} dB)")
    print(f"    Best FBP: {best_fbp} (PSNR={best_fbp_metrics['psnr']:.2f} dB)")
    print(f"    Best iterative: {best_iter} (PSNR={best_iter_metrics['psnr']:.2f} dB)")

    # ------------------------------------------------------------------
    # 5. Visualization: 4-panel figure
    # ------------------------------------------------------------------
    print("\n[5] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(phantom, cmap='gray', vmin=0, vmax=phantom.max())
    ax.set_title(f'(a) Ground Truth\n(Shepp-Logan Phantom {IMAGE_SIZE}×{IMAGE_SIZE})',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) Sinogram
    ax = axes[0, 1]
    im = ax.imshow(sinogram_noisy, cmap='hot', aspect='auto',
                   extent=[theta[0], theta[-1], sinogram_noisy.shape[0], 0])
    ax.set_title(f'(b) Noisy Sinogram\nSNR={actual_snr:.1f} dB, {NUM_ANGLES} angles',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Detector position')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) Best FBP reconstruction
    ax = axes[1, 0]
    disp = np.clip(best_fbp_recon, 0, phantom.max())
    im = ax.imshow(disp, cmap='gray', vmin=0, vmax=phantom.max())
    ax.set_title(f'(c) {best_fbp} Reconstruction\nPSNR={best_fbp_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_fbp_metrics["ssim"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) Best iterative reconstruction
    ax = axes[1, 1]
    disp = np.clip(best_iter_recon, 0, phantom.max())
    im = ax.imshow(disp, cmap='gray', vmin=0, vmax=phantom.max())
    ax.set_title(f'(d) {best_iter} Reconstruction\nPSNR={best_iter_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_iter_metrics["ssim"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('CT Reconstruction via Radon Transform Inversion\n'
                 f'Best: {best_method} — PSNR={best_metrics["psnr"]:.2f} dB, '
                 f'SSIM={best_metrics["ssim"]:.4f}',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved figure: {fig_path}")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print("\n[6] Saving outputs...")

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), phantom)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_recon)

    metrics = {
        "task": "torch_radon_ct",
        "inverse_problem": "CT reconstruction via Radon transform inversion",
        "image_size": IMAGE_SIZE,
        "num_angles": NUM_ANGLES,
        "noise_snr_db": float(actual_snr),
        "best_method": best_method,
        "best_psnr_db": float(best_metrics['psnr']),
        "best_ssim": float(best_metrics['ssim']),
        "best_rmse": float(best_metrics['rmse']),
        "all_methods": {
            method: {
                "psnr_db": float(v['psnr']),
                "ssim": float(v['ssim']),
                "rmse": float(v['rmse']),
                "time_seconds": float(v['time'])
            } for method, v in results.items()
        }
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("    Saved: ground_truth.npy, reconstruction.npy, metrics.json")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ground truth:   {IMAGE_SIZE}x{IMAGE_SIZE} Shepp-Logan phantom")
    print(f"  Forward model:  Radon transform, {NUM_ANGLES} angles, SNR={actual_snr:.1f} dB")
    print(f"  Methods compared:")
    for method, v in results.items():
        print(f"    {method:14s}  PSNR={v['psnr']:.2f} dB  SSIM={v['ssim']:.4f}  RMSE={v['rmse']:.4f}  t={v['time']:.1f}s")
    print(f"  Best method:    {best_method}")
    print(f"  Best PSNR:      {best_metrics['psnr']:.2f} dB")
    print(f"  Best SSIM:      {best_metrics['ssim']:.4f}")
    print("=" * 60)

    return best_metrics


if __name__ == "__main__":
    metrics = main()
