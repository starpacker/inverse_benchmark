"""
Task 191: flowdec_deconv
3D Richardson-Lucy Deconvolution for Fluorescence Microscopy Images

Inverse problem: Given a blurred + noisy 3D fluorescence microscopy volume,
recover the original clean volume using Richardson-Lucy (RL) deconvolution.

Pipeline:
  1. Generate synthetic 3D fluorescence volume (sparse bright blobs)
  2. Generate anisotropic 3D Gaussian PSF (wide in z, narrow in xy)
  3. Forward model: convolve volume with PSF + Poisson noise
  4. Inverse: Richardson-Lucy deconvolution (scikit-image)
  5. Evaluate: PSNR and SSIM on central z-slice
  6. Visualize: GT, blurred, deconvolved, error maps
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def make_results_dir():
    """Create results directory."""
    os.makedirs('results', exist_ok=True)


def generate_3d_fluorescence_volume(shape=(32, 128, 128), n_blobs=40, seed=42):
    """
    Generate a synthetic 3D fluorescence microscopy volume with bright blobs
    on a dark background, mimicking fluorescent bead or cell samples.

    Parameters
    ----------
    shape : tuple
        (nz, ny, nx) volume dimensions.
    n_blobs : int
        Number of fluorescent blobs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    volume : np.ndarray, shape (nz, ny, nx), float64 in [0, 1]
    """
    rng = np.random.RandomState(seed)
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.float64)

    for _ in range(n_blobs):
        # Random center within the volume (with margin)
        cz = rng.randint(2, nz - 2)
        cy = rng.randint(10, ny - 10)
        cx = rng.randint(10, nx - 10)
        # Random radius in z and xy
        rz = rng.uniform(1.0, 2.5)
        rxy = rng.uniform(2.0, 5.0)
        intensity = rng.uniform(0.5, 1.0)

        # Create coordinate grids relative to blob center
        zz, yy, xx = np.ogrid[
            max(0, int(cz - 4*rz)):min(nz, int(cz + 4*rz) + 1),
            max(0, int(cy - 4*rxy)):min(ny, int(cy + 4*rxy) + 1),
            max(0, int(cx - 4*rxy)):min(nx, int(cx + 4*rxy) + 1)
        ]
        blob = intensity * np.exp(
            -((zz - cz)**2 / (2 * rz**2)
              + (yy - cy)**2 / (2 * rxy**2)
              + (xx - cx)**2 / (2 * rxy**2))
        )
        volume[
            max(0, int(cz - 4*rz)):min(nz, int(cz + 4*rz) + 1),
            max(0, int(cy - 4*rxy)):min(ny, int(cy + 4*rxy) + 1),
            max(0, int(cx - 4*rxy)):min(nx, int(cx + 4*rxy) + 1)
        ] += blob

    # Add low-level background fluorescence
    volume += 0.02
    # Clip and normalize to [0, 1]
    volume = np.clip(volume, 0, None)
    volume /= volume.max()
    return volume


def generate_3d_psf(shape=(32, 128, 128), sigma_z=3.0, sigma_xy=1.5):
    """
    Generate an anisotropic 3D Gaussian PSF mimicking a widefield fluorescence
    microscope. The PSF is elongated along z (axial) and compact in xy (lateral).

    Parameters
    ----------
    shape : tuple
        (nz, ny, nx) — same as the volume for full-size convolution.
    sigma_z : float
        Standard deviation along z-axis (axial blur).
    sigma_xy : float
        Standard deviation along y and x axes (lateral blur).

    Returns
    -------
    psf : np.ndarray, normalized so sum = 1
    """
    nz, ny, nx = shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    psf = np.exp(
        -((zz - cz)**2 / (2 * sigma_z**2)
          + (yy - cy)**2 / (2 * sigma_xy**2)
          + (xx - cx)**2 / (2 * sigma_xy**2))
    )
    psf /= psf.sum()
    return psf


def forward_model(volume, psf, photon_count=1000, seed=42):
    """
    Forward model: convolve clean volume with PSF, then add Poisson noise
    to simulate photon-limited fluorescence microscopy acquisition.

    Parameters
    ----------
    volume : np.ndarray
        Clean 3D volume, values in [0, 1].
    psf : np.ndarray
        Point spread function (same shape as volume), normalized.
    photon_count : int
        Peak photon count controlling noise level.
    seed : int
        Random seed.

    Returns
    -------
    blurred_noisy : np.ndarray
        Blurred and noisy observation.
    blurred_clean : np.ndarray
        Blurred but noiseless (for reference).
    """
    rng = np.random.RandomState(seed)

    # Convolve with PSF using FFT-based convolution
    blurred = fftconvolve(volume, psf, mode='same')
    blurred = np.clip(blurred, 0, None)
    blurred_clean = blurred.copy()

    # Scale to photon counts and apply Poisson noise
    blurred_scaled = blurred * photon_count
    noisy_scaled = rng.poisson(np.clip(blurred_scaled, 0, None)).astype(np.float64)
    blurred_noisy = noisy_scaled / photon_count

    return blurred_noisy, blurred_clean


def inverse_richardson_lucy(observed, psf, n_iterations=80):
    """
    Richardson-Lucy deconvolution using scikit-image.

    The RL algorithm iteratively estimates the original image:
        x_{k+1} = x_k * (PSF^T ⊛ (y / (PSF ⊛ x_k)))

    Parameters
    ----------
    observed : np.ndarray
        Blurred + noisy 3D volume.
    psf : np.ndarray
        Point spread function.
    n_iterations : int
        Number of RL iterations.

    Returns
    -------
    deconvolved : np.ndarray
        Reconstructed 3D volume.
    """
    # Ensure non-negative input (RL requires positive values)
    observed_pos = np.clip(observed, 1e-12, None)

    # scikit-image richardson_lucy expects the PSF, not the full-size kernel
    # Extract the compact PSF kernel from the full-size array
    # Find the region with significant PSF values
    nz, ny, nx = psf.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    # Use a window large enough to capture the PSF
    wz, wy, wx = min(15, nz // 2), min(15, ny // 2), min(15, nx // 2)
    psf_compact = psf[
        cz - wz:cz + wz + 1,
        cy - wy:cy + wy + 1,
        cx - wx:cx + wx + 1
    ].copy()
    psf_compact /= psf_compact.sum()

    deconvolved = richardson_lucy(observed_pos, psf_compact, num_iter=n_iterations,
                                   clip=False)
    return deconvolved


def evaluate_metrics(gt, recon):
    """
    Compute PSNR and SSIM on the central z-slice.

    Parameters
    ----------
    gt : np.ndarray, 3D
    recon : np.ndarray, 3D

    Returns
    -------
    metrics : dict with 'psnr' and 'ssim'
    """
    mid_z = gt.shape[0] // 2

    gt_slice = gt[mid_z]
    recon_slice = recon[mid_z]

    # Normalize both to [0, 1] using GT range for fair comparison
    vmin, vmax = gt_slice.min(), gt_slice.max()
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0

    gt_norm = (gt_slice - vmin) / (vmax - vmin)
    recon_norm = np.clip((recon_slice - vmin) / (vmax - vmin), 0, 1)

    psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)

    return {'psnr': float(round(psnr_val, 4)), 'ssim': float(round(ssim_val, 4))}


def visualize_results(gt, observed, deconvolved, metrics, save_path='results/reconstruction_result.png'):
    """
    Visualize central z-slice: GT, blurred input, RL deconvolved, error map.
    """
    mid_z = gt.shape[0] // 2
    gt_s = gt[mid_z]
    obs_s = observed[mid_z]
    dec_s = deconvolved[mid_z]
    err_s = np.abs(gt_s - dec_s)

    vmin, vmax = gt_s.min(), gt_s.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(gt_s, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth (z-center)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(obs_s, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_title('Blurred + Noisy Input', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(dec_s, cmap='hot', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'RL Deconvolved\nPSNR={metrics["psnr"]:.2f} dB, SSIM={metrics["ssim"]:.4f}',
                       fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_s, cmap='viridis')
    axes[3].set_title('|GT − Deconvolved| Error', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('Task 191: 3D Richardson-Lucy Deconvolution (Fluorescence Microscopy)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    make_results_dir()

    # --- Volume parameters ---
    vol_shape = (32, 128, 128)   # 32 z-slices, 128×128 lateral
    n_blobs = 40
    sigma_z = 2.5                # Axial PSF spread (wider)
    sigma_xy = 1.2               # Lateral PSF spread (narrower)
    photon_count = 5000          # Photon budget (noise level)
    n_iterations = 60            # RL iterations

    print("=" * 60)
    print("Task 191: 3D Richardson-Lucy Deconvolution")
    print("=" * 60)

    # Step 1: Generate synthetic fluorescence volume
    print("\n[1/5] Generating synthetic 3D fluorescence volume...")
    gt_volume = generate_3d_fluorescence_volume(vol_shape, n_blobs=n_blobs, seed=42)
    print(f"  Volume shape: {gt_volume.shape}, range: [{gt_volume.min():.4f}, {gt_volume.max():.4f}]")

    # Step 2: Generate PSF
    print("\n[2/5] Generating anisotropic 3D Gaussian PSF...")
    psf = generate_3d_psf(vol_shape, sigma_z=sigma_z, sigma_xy=sigma_xy)
    print(f"  PSF shape: {psf.shape}, sigma_z={sigma_z}, sigma_xy={sigma_xy}")

    # Step 3: Forward model — blur + Poisson noise
    print("\n[3/5] Applying forward model (convolution + Poisson noise)...")
    observed, blurred_clean = forward_model(gt_volume, psf, photon_count=photon_count, seed=42)
    print(f"  Observed range: [{observed.min():.4f}, {observed.max():.4f}]")

    # Step 4: Inverse — Richardson-Lucy deconvolution
    print(f"\n[4/5] Running Richardson-Lucy deconvolution ({n_iterations} iterations)...")
    deconvolved = inverse_richardson_lucy(observed, psf, n_iterations=n_iterations)
    print(f"  Deconvolved range: [{deconvolved.min():.6f}, {deconvolved.max():.6f}]")

    # Step 5: Evaluate
    print("\n[5/5] Evaluating reconstruction quality...")
    metrics = evaluate_metrics(gt_volume, deconvolved)
    print(f"  PSNR: {metrics['psnr']:.4f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")

    # Also compute metrics for blurred input (baseline)
    metrics_blurred = evaluate_metrics(gt_volume, observed)
    print(f"\n  [Baseline] Blurred input — PSNR: {metrics_blurred['psnr']:.4f} dB, SSIM: {metrics_blurred['ssim']:.4f}")

    # Save metrics
    metrics_out = {
        'task': 'flowdec_deconv',
        'task_number': 191,
        'method': 'Richardson-Lucy deconvolution (scikit-image)',
        'n_iterations': n_iterations,
        'volume_shape': list(vol_shape),
        'psf_sigma_z': sigma_z,
        'psf_sigma_xy': sigma_xy,
        'photon_count': photon_count,
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'baseline_psnr': metrics_blurred['psnr'],
        'baseline_ssim': metrics_blurred['ssim']
    }
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print("\n  Metrics saved to results/metrics.json")

    # Save arrays
    np.save('results/ground_truth.npy', gt_volume)
    np.save('results/reconstruction.npy', deconvolved)
    print("  Arrays saved to results/")

    # Visualize
    visualize_results(gt_volume, observed, deconvolved, metrics)

    print("\n" + "=" * 60)
    print(f"DONE — PSNR: {metrics['psnr']:.4f} dB | SSIM: {metrics['ssim']:.4f}")
    print("=" * 60)

    return metrics


if __name__ == '__main__':
    main()
