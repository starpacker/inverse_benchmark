"""
ODTbrain Benchmark: 3D Refractive Index Reconstruction from Holographic Projections

Reconstructs a 3D refractive index distribution from simulated holographic
projections using Born/Rytov diffraction tomography backpropagation.

Reference: Müller et al., BMC Bioinformatics 2015
Library: ODTbrain (RI-imaging/ODTbrain)

Pipeline:
  1. Generate a 3D RI phantom (sphere with known refractive index)
  2. Simulate scattered field sinogram using the Born approximation forward model
  3. Add measurement noise
  4. Reconstruct 3D RI using Rytov backpropagation (odtbrain.backpropagate_3d)
  5. Compute metrics (PSNR, SSIM, RMSE) and save results
"""
import sys
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import rotate

# Add repo to path
sys.path.insert(0, '/data/yjh/odtbrain_sandbox/repo')
import odtbrain as odt

# ============================================================
# Output directory
# ============================================================
RESULTS_DIR = '/data/yjh/odtbrain_sandbox/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
# Grid size (cubic)
N = 48           # voxels per axis (keep moderate for speed)
# Number of projection angles
NUM_ANGLES = 160
# Wavelength in pixels (res = lambda / pixel_size)
RES = 5.0        # vacuum wavelength in pixels
# Refractive indices
NM = 1.335       # medium (water-like)
N_SPHERE = 1.345 # sphere (slightly higher than medium)
# Sphere radius in voxels
R_SPHERE = N // 6  # ~10 voxels
# Noise level (std of complex Gaussian noise relative to signal)
NOISE_LEVEL = 0.005
# Detector distance from rotation center (in pixels)
LD = 0.0

# ============================================================
# 1. Generate 3D RI Phantom
# ============================================================
def create_sphere_phantom(N, nm, n_sphere, radius):
    """Create a 3D sphere phantom with known refractive index."""
    phantom = np.full((N, N, N), nm, dtype=np.float64)
    center = N // 2
    zz, yy, xx = np.mgrid[:N, :N, :N]
    dist = np.sqrt((xx - center)**2 + (yy - center)**2 + (zz - center)**2)
    phantom[dist <= radius] = n_sphere
    return phantom

print("Step 1: Generating 3D RI phantom...")
phantom_ri = create_sphere_phantom(N, NM, N_SPHERE, R_SPHERE)
print(f"  Phantom shape: {phantom_ri.shape}")
print(f"  RI range: [{phantom_ri.min():.4f}, {phantom_ri.max():.4f}]")

# ============================================================
# 2. Compute object function f from RI
# ============================================================
# f(r) = k_m^2 * [(n(r)/n_m)^2 - 1]
km = 2 * np.pi * NM / RES
f_obj = km**2 * ((phantom_ri / NM)**2 - 1)
print(f"  Object function range: [{f_obj.min():.6f}, {f_obj.max():.6f}]")

# ============================================================
# 3. Simulate Sinogram (Born Approximation Forward Model)
# ============================================================
def simulate_born_sinogram(f_obj, angles, res, nm, N):
    """
    Simulate a scattered field sinogram using the first-order Born
    approximation. For each angle, compute the projection of the object
    function through the rotated volume, then convert to a complex
    scattered field.

    The Born scattered field at the detector is approximately:
        u_B(x,y) ~ integral of f(r) * exp(i*km*z) dz
    For weak scattering, the total field is:
        u_total = u_0 + u_B ≈ 1 + u_B (for unit incident field)

    We use the projection-slice approach: rotate the object function,
    integrate along z, and compute the scattered field.
    """
    km_val = 2 * np.pi * nm / res
    num_angles = len(angles)
    # Sinogram: (A, Ny, Nx) complex array
    sino = np.zeros((num_angles, N, N), dtype=np.complex128)

    for i, angle in enumerate(angles):
        # Rotate the object function about the y-axis (axis=1)
        # ODTbrain convention: rotation about y-axis, in the xz plane
        angle_deg = np.degrees(angle)
        # Rotate f_obj: axes (0,2) means rotation in the xz plane
        f_rotated = rotate(f_obj, angle=-angle_deg, axes=(0, 2),
                           reshape=False, order=1, mode='constant', cval=0)

        # Project along z-axis (axis=2 after rotation = propagation direction)
        # The Born scattered field: u_B(x,y) ~ (i/(2*km)) * integral f(r) dz
        # (simplified; actual Green's function integral is more complex,
        #  but for the first-order Born this gives the phase/amplitude modulation)
        projection = np.sum(f_rotated, axis=2).astype(np.complex128)

        # Scale by voxel size (1 pixel) and the Born kernel factor
        # u_B = (i / (2*km)) * projection
        u_B = (1j / (2 * km_val)) * projection

        # Total field = incident + scattered
        # u_total = u_0 * (1 + u_B/u_0) where u_0 = 1
        sino[i] = 1.0 + u_B

    return sino

print("\nStep 2: Simulating Born sinogram...")
angles = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)
sino = simulate_born_sinogram(f_obj, angles, RES, NM, N)
print(f"  Sinogram shape: {sino.shape}")
print(f"  Sinogram amplitude range: [{np.abs(sino).min():.4f}, {np.abs(sino).max():.4f}]")
print(f"  Sinogram phase range: [{np.angle(sino).min():.4f}, {np.angle(sino).max():.4f}]")

# ============================================================
# 4. Add Noise
# ============================================================
print("\nStep 3: Adding measurement noise...")
np.random.seed(42)
noise = NOISE_LEVEL * (np.random.randn(*sino.shape) + 1j * np.random.randn(*sino.shape))
sino_noisy = sino + noise
print(f"  Noise level: {NOISE_LEVEL}")
print(f"  SNR (approx): {10*np.log10(np.mean(np.abs(sino)**2) / np.mean(np.abs(noise)**2)):.1f} dB")

# ============================================================
# 5. Reconstruct using Rytov Backpropagation
# ============================================================
print("\nStep 4: Applying Rytov approximation and backpropagation...")

# Convert sinogram to Rytov phase
sino_rytov = odt.sinogram_as_rytov(sino_noisy)
print(f"  Rytov sinogram shape: {sino_rytov.shape}")
print(f"  Rytov sinogram range: [{sino_rytov.real.min():.4f}, {sino_rytov.real.max():.4f}]")

# 3D Backpropagation
f_recon = odt.backpropagate_3d(
    uSin=sino_rytov,
    angles=angles,
    res=RES,
    nm=NM,
    lD=LD,
    padfac=1.75,
    padding=(True, True),
    padval="edge",
    onlyreal=False,
    intp_order=2,
    save_memory=True,
    num_cores=1,
)
print(f"  Reconstructed object function shape: {f_recon.shape}")

# Convert object function to refractive index
ri_recon = odt.odt_to_ri(f_recon, res=RES, nm=NM)
print(f"  Reconstructed RI shape: {ri_recon.shape}")
print(f"  Reconstructed RI range: [{ri_recon.real.min():.4f}, {ri_recon.real.max():.4f}]")

# ============================================================
# 6. Align volumes (reconstruction may be different size due to padding)
# ============================================================
# Crop or pad to match phantom size
recon_shape = ri_recon.shape
phantom_shape = phantom_ri.shape

if recon_shape != phantom_shape:
    print(f"\nStep 5: Aligning reconstruction ({recon_shape}) to phantom ({phantom_shape})...")
    # Center-crop the reconstruction
    ri_aligned = np.full(phantom_shape, NM, dtype=np.complex128)
    slices_src = []
    slices_dst = []
    for i in range(3):
        r = recon_shape[i]
        p = phantom_shape[i]
        if r >= p:
            start = (r - p) // 2
            slices_src.append(slice(start, start + p))
            slices_dst.append(slice(0, p))
        else:
            start = (p - r) // 2
            slices_src.append(slice(0, r))
            slices_dst.append(slice(start, start + r))
    ri_aligned[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        ri_recon[slices_src[0], slices_src[1], slices_src[2]]
    ri_recon_aligned = ri_aligned.real
else:
    ri_recon_aligned = ri_recon.real

print(f"  Aligned RI shape: {ri_recon_aligned.shape}")
print(f"  Aligned RI range: [{ri_recon_aligned.min():.4f}, {ri_recon_aligned.max():.4f}]")

# ============================================================
# 7. Compute Metrics
# ============================================================
print("\nStep 6: Computing metrics...")

def compute_psnr(gt, recon):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float('inf')
    data_range = gt.max() - gt.min()
    return 10 * np.log10(data_range ** 2 / mse)

def compute_ssim(gt, recon, data_range=None):
    """Compute SSIM for 2D images."""
    from skimage.metrics import structural_similarity
    if data_range is None:
        data_range = gt.max() - gt.min()
    return structural_similarity(gt, recon, data_range=data_range)

def compute_rmse(gt, recon):
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((gt - recon) ** 2))

# Use central slices for 2D metrics
center = N // 2
gt_slice = phantom_ri[center, :, :]          # central xz slice
recon_slice = ri_recon_aligned[center, :, :]  # central xz slice

# Also compute on other central slices
gt_slice_yz = phantom_ri[:, center, :]
recon_slice_yz = ri_recon_aligned[:, center, :]
gt_slice_xy = phantom_ri[:, :, center]
recon_slice_xy = ri_recon_aligned[:, :, center]

data_range = phantom_ri.max() - phantom_ri.min()

# 2D metrics on central slice
psnr_2d = compute_psnr(gt_slice, recon_slice)
ssim_2d = compute_ssim(gt_slice, recon_slice, data_range=data_range)
rmse_2d = compute_rmse(gt_slice, recon_slice)

# 3D metrics (volumetric)
psnr_3d = compute_psnr(phantom_ri, ri_recon_aligned)
ssim_slices = []
for i in range(phantom_ri.shape[0]):
    s = compute_ssim(phantom_ri[i], ri_recon_aligned[i], data_range=data_range)
    ssim_slices.append(s)
ssim_3d = float(np.mean(ssim_slices))
rmse_3d = compute_rmse(phantom_ri, ri_recon_aligned)

print(f"  Central slice - PSNR: {psnr_2d:.2f} dB, SSIM: {ssim_2d:.4f}, RMSE: {rmse_2d:.6f}")
print(f"  3D volume     - PSNR: {psnr_3d:.2f} dB, SSIM: {ssim_3d:.4f}, RMSE: {rmse_3d:.6f}")

metrics = {
    'PSNR': round(float(psnr_2d), 2),
    'SSIM': round(float(ssim_2d), 4),
    'RMSE': round(float(rmse_2d), 6),
    'PSNR_3D': round(float(psnr_3d), 2),
    'SSIM_3D': round(float(ssim_3d), 4),
    'RMSE_3D': round(float(rmse_3d), 6),
    'num_projections': NUM_ANGLES,
    'grid_size': N,
    'noise_level': NOISE_LEVEL,
    'medium_index': NM,
    'sphere_index': N_SPHERE,
    'method': 'Rytov backpropagation (ODTbrain)'
}

metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved to {metrics_path}")

# ============================================================
# 8. Visualization
# ============================================================
print("\nStep 7: Creating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Ground truth, Sinogram, Reconstruction
# GT central slice
im0 = axes[0, 0].imshow(gt_slice, vmin=NM-0.001, vmax=N_SPHERE+0.001,
                          cmap='hot', interpolation='none')
axes[0, 0].set_title('Ground Truth (central xz slice)', fontsize=11)
axes[0, 0].set_xlabel('x [px]')
axes[0, 0].set_ylabel('z [px]')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, format='%.4f')

# Sinogram (phase of one projection)
sino_phase = np.angle(sino_noisy)
im1 = axes[0, 1].imshow(sino_phase[:, center, :],
                          aspect=sino.shape[2]/sino.shape[0],
                          cmap='coolwarm', interpolation='none')
axes[0, 1].set_title('Phase Sinogram (y=center)', fontsize=11)
axes[0, 1].set_xlabel('detector x [px]')
axes[0, 1].set_ylabel('angle index')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# Reconstruction central slice
im2 = axes[0, 2].imshow(recon_slice, vmin=NM-0.001, vmax=N_SPHERE+0.001,
                          cmap='hot', interpolation='none')
axes[0, 2].set_title(f'Reconstruction (PSNR={psnr_2d:.1f}dB, SSIM={ssim_2d:.3f})',
                      fontsize=11)
axes[0, 2].set_xlabel('x [px]')
axes[0, 2].set_ylabel('z [px]')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, format='%.4f')

# Row 2: Error map, other slices
# Error map
error = np.abs(gt_slice - recon_slice)
im3 = axes[1, 0].imshow(error, cmap='hot', interpolation='none')
axes[1, 0].set_title('Absolute Error Map', fontsize=11)
axes[1, 0].set_xlabel('x [px]')
axes[1, 0].set_ylabel('z [px]')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, format='%.5f')

# Line profile through center
center_line_gt = phantom_ri[center, center, :]
center_line_recon = ri_recon_aligned[center, center, :]
axes[1, 1].plot(center_line_gt, 'b-', linewidth=2, label='Ground Truth')
axes[1, 1].plot(center_line_recon, 'r--', linewidth=2, label='Reconstruction')
axes[1, 1].set_title('Line Profile (through center)', fontsize=11)
axes[1, 1].set_xlabel('x [px]')
axes[1, 1].set_ylabel('Refractive Index')
axes[1, 1].legend()
axes[1, 1].set_ylim([NM - 0.002, N_SPHERE + 0.002])

# YZ reconstruction slice
im5 = axes[1, 2].imshow(ri_recon_aligned[:, :, center].T,
                          vmin=NM-0.001, vmax=N_SPHERE+0.001,
                          cmap='hot', interpolation='none')
axes[1, 2].set_title('Reconstruction (yz slice at x=center)', fontsize=11)
axes[1, 2].set_xlabel('z [px]')
axes[1, 2].set_ylabel('y [px]')
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, format='%.4f')

plt.suptitle('ODTbrain: 3D RI Reconstruction via Rytov Backpropagation\n'
             f'N={N}, {NUM_ANGLES} projections, noise={NOISE_LEVEL}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure saved to {fig_path}")

# ============================================================
# 9. Save numpy arrays
# ============================================================
print("\nStep 8: Saving numpy arrays...")
gt_path = os.path.join(RESULTS_DIR, 'ground_truth.npy')
recon_path = os.path.join(RESULTS_DIR, 'reconstruction.npy')
np.save(gt_path, phantom_ri)
np.save(recon_path, ri_recon_aligned)
print(f"  Ground truth saved to {gt_path}")
print(f"  Reconstruction saved to {recon_path}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("RECONSTRUCTION COMPLETE")
print("=" * 60)
print(f"  Method: Rytov backpropagation (ODTbrain)")
print(f"  Grid: {N}x{N}x{N}")
print(f"  Projections: {NUM_ANGLES}")
print(f"  Noise level: {NOISE_LEVEL}")
print(f"  PSNR (2D central): {psnr_2d:.2f} dB")
print(f"  SSIM (2D central): {ssim_2d:.4f}")
print(f"  RMSE (2D central): {rmse_2d:.6f}")
print(f"  PSNR (3D volume):  {psnr_3d:.2f} dB")
print(f"  SSIM (3D volume):  {ssim_3d:.4f}")
print(f"  RMSE (3D volume):  {rmse_3d:.6f}")
print("=" * 60)
