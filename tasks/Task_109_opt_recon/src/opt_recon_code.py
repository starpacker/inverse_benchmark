"""
Task 119: Optical Projection Tomography (OPT) 3D Reconstruction

OPT is essentially optical CT — it acquires projection images of a transparent
sample at multiple angles and uses tomographic reconstruction (FBP) to recover
the 3D volume. This script:

1. Generates a synthetic 3D phantom (64x128x128) with known geometric features
2. Forward operator: Radon transform (parallel-beam projections) at multiple angles
3. Adds Poisson + Gaussian readout noise
4. Reconstructs via Filtered Back-Projection (FBP) using skimage.transform.iradon
5. Evaluates: PSNR, SSIM, RMSE
6. Visualizes: GT slice, sinogram, reconstruction slice, error map
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def create_3d_phantom(nz=64, ny=128, nx=128):
    """
    Create a synthetic 3D phantom for OPT reconstruction.
    The phantom contains:
    - A large outer ellipsoid (simulating a tissue sample)
    - Several internal structures (spheres, cylinders) with varying intensities
    """
    phantom = np.zeros((nz, ny, nx), dtype=np.float64)

    cz, cy, cx = nz // 2, ny // 2, nx // 2

    # Create coordinate grids
    z, y, x = np.ogrid[:nz, :ny, :nx]

    # Outer ellipsoid (normalized coords)
    outer = ((z - cz) / (cz * 0.85))**2 + ((y - cy) / (cy * 0.8))**2 + ((x - cx) / (cx * 0.8))**2
    phantom[outer <= 1.0] = 0.5

    # Inner sphere 1 (high intensity, offset)
    r1 = ((z - cz) / 12.0)**2 + ((y - (cy - 20)) / 15.0)**2 + ((x - (cx + 15)) / 15.0)**2
    phantom[r1 <= 1.0] = 1.0

    # Inner sphere 2 (medium intensity, opposite side)
    r2 = ((z - cz) / 10.0)**2 + ((y - (cy + 25)) / 12.0)**2 + ((x - (cx - 20)) / 12.0)**2
    phantom[r2 <= 1.0] = 0.8

    # Small dense sphere near center
    r3 = ((z - cz) / 6.0)**2 + ((y - cy) / 8.0)**2 + ((x - cx) / 8.0)**2
    phantom[r3 <= 1.0] = 0.9

    # A rod/cylinder along z-axis (off-center)
    rod = ((y - (cy + 10)) / 5.0)**2 + ((x - (cx - 30)) / 5.0)**2
    phantom[:, rod[0, :, :] <= 1.0] = 0.7

    # Another small feature
    r4 = ((z - (cz + 15)) / 8.0)**2 + ((y - (cy - 15)) / 6.0)**2 + ((x - (cx + 30)) / 6.0)**2
    phantom[r4 <= 1.0] = 0.6

    return phantom


def forward_operator(phantom_slice, theta):
    """
    Compute the Radon transform (sinogram) of a 2D slice.
    This simulates parallel-beam optical projections at given angles.
    """
    sinogram = radon(phantom_slice, theta=theta, circle=True)
    return sinogram


def add_noise(sinogram, photon_count=1e4, readout_std=0.5):
    """
    Add realistic OPT noise:
    - Poisson noise (photon-limited imaging)
    - Gaussian readout noise
    """
    sino_min = sinogram.min()
    sino_max = sinogram.max()
    if sino_max - sino_min < 1e-10:
        return sinogram.copy()

    sino_norm = (sinogram - sino_min) / (sino_max - sino_min)

    # Poisson noise: scale to photon counts, sample, scale back
    sino_photons = sino_norm * photon_count
    sino_noisy = np.random.poisson(sino_photons).astype(np.float64) / photon_count

    # Readout noise (Gaussian)
    sino_noisy += np.random.normal(0, readout_std / photon_count, sino_noisy.shape)

    # Scale back to original range
    sino_noisy = sino_noisy * (sino_max - sino_min) + sino_min

    return sino_noisy


def reconstruct_fbp(sinogram, theta):
    """
    Filtered Back-Projection reconstruction using skimage.
    Uses the Ram-Lak (ramp) filter.
    """
    recon = iradon(sinogram, theta=theta, circle=True, filter_name='ramp')
    return recon


def compute_metrics(gt, recon):
    """Compute PSNR, SSIM, and RMSE between ground truth and reconstruction."""
    data_range = gt.max() - gt.min()
    if data_range < 1e-10:
        data_range = 1.0

    psnr = peak_signal_noise_ratio(gt, recon, data_range=data_range)
    ssim = structural_similarity(gt, recon, data_range=data_range)
    rmse = np.sqrt(np.mean((gt - recon) ** 2))

    return psnr, ssim, rmse


def main():
    np.random.seed(42)

    # Parameters
    nz, ny, nx = 64, 128, 128
    n_angles = 180  # number of projection angles (0 to 180 degrees)
    theta = np.linspace(0., 180., n_angles, endpoint=False)
    photon_count = 5e4  # higher photon count = less noise
    readout_std = 0.3

    print("=" * 60)
    print("Task 119: Optical Projection Tomography (OPT) Reconstruction")
    print("=" * 60)

    # Step 1: Generate 3D phantom
    print("\n[1/5] Generating 3D phantom...")
    phantom = create_3d_phantom(nz, ny, nx)
    print(f"  Phantom shape: {phantom.shape}")
    print(f"  Value range: [{phantom.min():.3f}, {phantom.max():.3f}]")

    # Step 2: Forward projection (slice-wise Radon transform)
    print("\n[2/5] Computing forward projections (sinograms)...")
    # Get sinogram size from a test projection
    test_sino = radon(phantom[0], theta=theta, circle=True)
    sino_height = test_sino.shape[0]
    sinograms = np.zeros((nz, sino_height, n_angles))
    for z in range(nz):
        sinograms[z] = forward_operator(phantom[z], theta)
    print(f"  Sinogram stack shape: {sinograms.shape}")

    # Step 3: Add noise
    print("\n[3/5] Adding Poisson + readout noise...")
    sinograms_noisy = np.zeros_like(sinograms)
    for z in range(nz):
        sinograms_noisy[z] = add_noise(sinograms[z], photon_count=photon_count,
                                        readout_std=readout_std)

    # Step 4: Reconstruct via FBP (slice-wise)
    print("\n[4/5] Reconstructing via Filtered Back-Projection (FBP)...")
    reconstruction = np.zeros((nz, ny, nx), dtype=np.float64)
    for z in range(nz):
        recon_slice = reconstruct_fbp(sinograms_noisy[z], theta)
        # iradon may return different size, crop/pad to match
        rh, rw = recon_slice.shape
        # Center-crop or center-pad to (ny, nx)
        sy = max(0, (rh - ny) // 2)
        sx = max(0, (rw - nx) // 2)
        dy = max(0, (ny - rh) // 2)
        dx = max(0, (nx - rw) // 2)
        h_copy = min(rh, ny)
        w_copy = min(rw, nx)
        reconstruction[z, dy:dy+h_copy, dx:dx+w_copy] = recon_slice[sy:sy+h_copy, sx:sx+w_copy]

    print(f"  Reconstruction shape: {reconstruction.shape}")

    # Step 5: Evaluate metrics on the full 3D volume
    print("\n[5/5] Computing evaluation metrics...")
    psnr_3d, ssim_3d, rmse_3d = compute_metrics(phantom, reconstruction)

    # Also compute per-slice metrics for the middle slice
    mid_z = nz // 2
    psnr_mid, ssim_mid, rmse_mid = compute_metrics(phantom[mid_z], reconstruction[mid_z])

    print(f"\n  3D Volume Metrics:")
    print(f"    PSNR: {psnr_3d:.2f} dB")
    print(f"    SSIM: {ssim_3d:.4f}")
    print(f"    RMSE: {rmse_3d:.6f}")
    print(f"\n  Middle Slice (z={mid_z}) Metrics:")
    print(f"    PSNR: {psnr_mid:.2f} dB")
    print(f"    SSIM: {ssim_mid:.4f}")
    print(f"    RMSE: {rmse_mid:.6f}")

    # Save results
    os.makedirs('results', exist_ok=True)

    metrics = {
        'task': 'opt_recon',
        'task_number': 119,
        'method': 'Filtered Back-Projection (FBP)',
        'inverse_problem': 'Optical Projection Tomography (OPT) 3D Reconstruction',
        'phantom_shape': list(phantom.shape),
        'n_angles': n_angles,
        'photon_count': photon_count,
        'readout_noise_std': readout_std,
        'metrics_3d': {
            'PSNR_dB': round(psnr_3d, 2),
            'SSIM': round(ssim_3d, 4),
            'RMSE': round(rmse_3d, 6)
        },
        'metrics_middle_slice': {
            'PSNR_dB': round(psnr_mid, 2),
            'SSIM': round(ssim_mid, 4),
            'RMSE': round(rmse_mid, 6)
        }
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n  Saved: results/metrics.json")

    np.save('results/ground_truth.npy', phantom)
    np.save('results/reconstruction.npy', reconstruction)
    print("  Saved: results/ground_truth.npy")
    print("  Saved: results/reconstruction.npy")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Task 119: OPT Reconstruction (Filtered Back-Projection)',
                 fontsize=16, fontweight='bold')

    gt_slice = phantom[mid_z]
    sino_slice = sinograms_noisy[mid_z]
    recon_slice = reconstruction[mid_z]
    error_map = np.abs(gt_slice - recon_slice)

    vmin, vmax = 0, gt_slice.max()

    # GT slice
    im0 = axes[0, 0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Ground Truth (z={mid_z})', fontsize=13)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Sinogram
    im1 = axes[0, 1].imshow(sino_slice.T, cmap='gray', aspect='auto',
                             extent=[0, sino_slice.shape[0], 180, 0])
    axes[0, 1].set_title(f'Sinogram (noisy, z={mid_z})', fontsize=13)
    axes[0, 1].set_xlabel('Detector position')
    axes[0, 1].set_ylabel('Angle (degrees)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Reconstruction
    im2 = axes[0, 2].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'FBP Reconstruction (z={mid_z})', fontsize=13)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Error map
    im3 = axes[1, 0].imshow(error_map, cmap='hot')
    axes[1, 0].set_title('Absolute Error Map', fontsize=13)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Line profile comparison
    mid_y = ny // 2
    axes[1, 1].plot(gt_slice[mid_y, :], 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(recon_slice[mid_y, :], 'r--', linewidth=2, label='FBP Recon')
    axes[1, 1].set_title(f'Line Profile (y={mid_y})', fontsize=13)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    # 3D overview: show multiple slices
    slice_indices = [nz // 4, nz // 2, 3 * nz // 4]
    for i, zi in enumerate(slice_indices):
        color = ['blue', 'green', 'red'][i]
        psnr_i, ssim_i, _ = compute_metrics(phantom[zi], reconstruction[zi])
        axes[1, 2].plot(reconstruction[zi][ny // 2, :], color=color, alpha=0.7,
                        label=f'z={zi} (PSNR={psnr_i:.1f}dB)')
    axes[1, 2].set_title('Reconstruction Profiles at Different z-Slices', fontsize=13)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)

    # Add metrics text
    fig.text(0.5, 0.01,
             f'3D Volume --- PSNR: {psnr_3d:.2f} dB | SSIM: {ssim_3d:.4f} | RMSE: {rmse_3d:.6f}',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig('results/reconstruction_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/reconstruction_result.png")

    print("\n" + "=" * 60)
    print("OPT Reconstruction Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
