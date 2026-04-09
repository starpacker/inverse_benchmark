"""
Task 126: prysm Phase Retrieval for Optical Systems

Phase retrieval: recover the pupil phase phi(x,y) from intensity measurements
I(x,y) = |F{P(x,y) * exp(j*phi(x,y))}|^2

Uses parametric phase diversity approach:
- Phase = sum of Zernike coefficients (parametric representation)
- Two PSF measurements at different known defocus settings
- Optimization via scipy.optimize.minimize (L-BFGS-B) to match
  model PSFs to measured PSFs in amplitude domain

This is more robust than pixel-wise Gerchberg-Saxton because:
1. The Zernike parametrization heavily constrains the solution space
2. Phase diversity breaks the sign ambiguity
3. Direct optimization avoids GS stagnation issues

Relies on prysm for Zernike polynomials and pupil geometry.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.polynomials import zernike_nm, zernike_nm_sequence, sum_of_2d_modes

# ============================================================
# Configuration
# ============================================================
NPIX = 128               # pupil array size (smaller for optimization speed)
WAVELENGTH = 0.6328      # HeNe wavelength, microns
EPD = 25.0               # entrance pupil diameter, mm
Q = 2                    # oversampling factor for PSF
PHOTON_FLUX = 5e6        # photon count
READOUT_NOISE = 3.0      # readout noise std
DEFOCUS_WAVES = 1.5      # diversity defocus in waves
RNG_SEED = 42

np.random.seed(RNG_SEED)

# Ensure we work in the sandbox directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
os.makedirs('results', exist_ok=True)

# ============================================================
# 1. Coordinate grids and pupil
# ============================================================
x, y = make_xy_grid(NPIX, diameter=EPD)
dx = x[0, 1] - x[0, 0]
r, t = cart_to_polar(x, y)

pupil_radius = EPD / 2.0
pupil_mask = circle(pupil_radius, r).astype(float)
pupil_bool = pupil_mask > 0.5

r_norm = r / pupil_radius

# ============================================================
# 2. Ground truth phase (Zernike aberrations)
# ============================================================
zernike_specs = [
    # (n, m, coeff_waves) -- moderate aberrations
    (2,  0,  0.20),   # Defocus
    (2,  2,  0.15),   # Astigmatism 0 deg
    (2, -2,  0.10),   # Astigmatism 45 deg
    (3,  1,  0.12),   # Coma X
    (3, -1,  0.08),   # Coma Y
    (3,  3,  0.05),   # Trefoil X
    (4,  0,  0.10),   # Primary Spherical
]

nms_truth = [(n, m) for n, m, _ in zernike_specs]
coefs_truth_waves = np.array([c for _, _, c in zernike_specs])

basis_truth = list(zernike_nm_sequence(nms_truth, r_norm, t, norm=True))
basis_truth = np.array(basis_truth)

true_phase_waves = sum_of_2d_modes(basis_truth, coefs_truth_waves)
true_phase_waves *= pupil_mask
true_phase_rad = true_phase_waves * 2 * np.pi

print(f"True phase: PV = {np.ptp(true_phase_rad[pupil_bool]):.3f} rad "
      f"({np.ptp(true_phase_waves[pupil_bool]):.3f} waves)")
print(f"True phase RMS = {np.std(true_phase_rad[pupil_bool]):.3f} rad "
      f"({np.std(true_phase_waves[pupil_bool]):.3f} waves)")

# ============================================================
# 3. Diversity defocus (known)
# ============================================================
defocus_mode = zernike_nm(2, 0, r_norm, t, norm=True)
diversity_phase_rad = DEFOCUS_WAVES * defocus_mode * pupil_mask * 2 * np.pi

# ============================================================
# 4. Forward model
# ============================================================
N_pad = NPIX * Q
pad_offset = (N_pad - NPIX) // 2


def make_psf(phase_rad):
    """Generate PSF from phase (uses module-level amplitude/padding)."""
    E_pupil = pupil_mask * np.exp(1j * phase_rad)
    E_pad = np.zeros((N_pad, N_pad), dtype=complex)
    E_pad[pad_offset:pad_offset+NPIX, pad_offset:pad_offset+NPIX] = E_pupil
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pad), norm='ortho'))
    return np.abs(E_focal)**2


def add_noise(psf, photon_flux, readout_noise):
    """Add Poisson + readout noise."""
    psf_photons = psf / (psf.sum() + 1e-30) * photon_flux
    noisy = np.random.poisson(np.clip(psf_photons, 0, None).astype(float)).astype(float)
    noisy += np.random.normal(0, readout_noise, noisy.shape)
    return np.clip(noisy, 0, None)


# Generate measurements
psf_infocus = make_psf(true_phase_rad)
psf_infocus_noisy = add_noise(psf_infocus, PHOTON_FLUX, READOUT_NOISE)

psf_defocus = make_psf(true_phase_rad + diversity_phase_rad)
psf_defocus_noisy = add_noise(psf_defocus, PHOTON_FLUX, READOUT_NOISE)

# Normalize measurements
total_1 = psf_infocus_noisy.sum()
total_2 = psf_defocus_noisy.sum()
psf_infocus_norm = psf_infocus_noisy / total_1
psf_defocus_norm = psf_defocus_noisy / total_2

print(f"\nPSF shape: {psf_infocus.shape}")
print(f"In-focus SNR ~ {np.sqrt(PHOTON_FLUX):.0f}")

# ============================================================
# 5. Retrieval Zernike basis
# ============================================================
retrieval_nms = []
for n in range(2, 6):
    for m in range(-n, n+1, 2):
        retrieval_nms.append((n, m))

n_modes = len(retrieval_nms)
print(f"\nRetrieval: {n_modes} Zernike modes (n=2..5)")

retrieval_basis = list(zernike_nm_sequence(retrieval_nms, r_norm, t, norm=True))
retrieval_basis = np.array(retrieval_basis)


def coefs_to_phase(coefs_waves):
    """Convert Zernike coefficients (in waves) to phase map (radians)."""
    phase_waves = sum_of_2d_modes(retrieval_basis, coefs_waves)
    return phase_waves * pupil_mask * 2 * np.pi


# ============================================================
# 6. Optimization-based phase retrieval
# ============================================================
call_count = [0]


def objective(coefs_waves):
    """
    Cost function: sum of squared differences between model and measured
    PSF amplitudes (sqrt of intensities) for both diversity channels.
    """
    phase_rad = coefs_to_phase(coefs_waves)

    # Channel 1: in-focus
    psf_m1 = make_psf(phase_rad)
    psf_m1_norm = psf_m1 / (psf_m1.sum() + 1e-30)
    err1 = np.sum((np.sqrt(psf_m1_norm) - np.sqrt(psf_infocus_norm))**2)

    # Channel 2: defocused
    psf_m2 = make_psf(phase_rad + diversity_phase_rad)
    psf_m2_norm = psf_m2 / (psf_m2.sum() + 1e-30)
    err2 = np.sum((np.sqrt(psf_m2_norm) - np.sqrt(psf_defocus_norm))**2)

    call_count[0] += 1
    return err1 + err2


def gradient(coefs_waves, eps=5e-4):
    """Finite-difference gradient."""
    grad = np.zeros(n_modes)
    f0 = objective(coefs_waves)
    for i in range(n_modes):
        cp = coefs_waves.copy()
        cp[i] += eps
        grad[i] = (objective(cp) - f0) / eps
    return grad


print("\nStarting optimization-based phase retrieval...")
print("  Phase 1: Coarse search with L-BFGS-B...")

# Initial guess: zero aberrations
x0 = np.zeros(n_modes)

# Multi-start: try a few random initial conditions
best_result = None
best_cost = float('inf')

for trial in range(5):
    if trial == 0:
        x_init = np.zeros(n_modes)
    else:
        x_init = np.random.randn(n_modes) * 0.1

    result = minimize(
        objective,
        x_init,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8},
    )

    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result
        print(f"  Trial {trial}: cost = {result.fun:.6e} (best so far)")
    else:
        print(f"  Trial {trial}: cost = {result.fun:.6e}")

coefs_opt = best_result.x
print(f"\n  Best cost after coarse: {best_cost:.6e}")

# Phase 2: Fine refinement
print("  Phase 2: Fine refinement...")
result_fine = minimize(
    objective,
    coefs_opt,
    method='L-BFGS-B',
    jac=lambda c: gradient(c, eps=1e-5),
    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10},
)
coefs_opt = result_fine.x
print(f"  Final cost: {result_fine.fun:.6e}")
print(f"  Total function evaluations: {call_count[0]}")

# Retrieved phase
retrieved_phase_rad = coefs_to_phase(coefs_opt)

# Print coefficient comparison
print(f"\n{'Mode':<12} {'True (waves)':<14} {'Retrieved (waves)':<18} {'Error (waves)':<14}")
print("-" * 58)

# Map truth modes to retrieval modes
truth_dict = {(n, m): c for n, m, c in zernike_specs}
for i, (n, m) in enumerate(retrieval_nms):
    true_c = truth_dict.get((n, m), 0.0)
    retr_c = coefs_opt[i]
    err_c = retr_c - true_c
    name = f"Z({n},{m:+d})"
    if abs(true_c) > 0 or abs(retr_c) > 0.01:
        print(f"  {name:<10} {true_c:>12.4f}   {retr_c:>14.4f}     {err_c:>12.4f}")

# ============================================================
# 7. Metrics
# ============================================================
true_mean = np.sum(true_phase_rad * pupil_mask) / np.sum(pupil_mask)
retr_mean = np.sum(retrieved_phase_rad * pupil_mask) / np.sum(pupil_mask)

true_centered = (true_phase_rad - true_mean) * pupil_mask
retrieved_centered = (retrieved_phase_rad - retr_mean) * pupil_mask

error_map = (retrieved_centered - true_centered) * pupil_mask

phase_rmse_rad = np.sqrt(np.mean(error_map[pupil_bool]**2))
phase_rmse_waves = phase_rmse_rad / (2 * np.pi)

signal_range = np.ptp(true_centered[pupil_bool])
phase_psnr = 20 * np.log10(signal_range / phase_rmse_rad) if phase_rmse_rad > 0 else float('inf')

cc = np.corrcoef(true_centered[pupil_bool], retrieved_centered[pupil_bool])[0, 1]

true_rms = np.std(true_centered[pupil_bool])
retr_rms = np.std(retrieved_centered[pupil_bool])
strehl_true = np.exp(-true_rms**2)
strehl_retrieved = np.exp(-retr_rms**2)

from skimage.metrics import structural_similarity as ssim
vmin = min(true_centered[pupil_bool].min(), retrieved_centered[pupil_bool].min())
vmax = max(true_centered[pupil_bool].max(), retrieved_centered[pupil_bool].max())
drange = vmax - vmin if (vmax - vmin) > 1e-10 else 1.0
true_norm = (true_centered - vmin) / drange * pupil_mask
retr_norm = (retrieved_centered - vmin) / drange * pupil_mask
ssim_val = ssim(true_norm, retr_norm, data_range=1.0)

print(f"\n{'='*50}")
print(f"Phase Retrieval Results:")
print(f"{'='*50}")
print(f"Phase RMSE:      {phase_rmse_rad:.4f} rad ({phase_rmse_waves:.4f} waves)")
print(f"Phase PSNR:      {phase_psnr:.2f} dB")
print(f"SSIM:            {ssim_val:.4f}")
print(f"Correlation:     {cc:.6f}")
print(f"Strehl (true):   {strehl_true:.6f}")
print(f"Strehl (retr):   {strehl_retrieved:.6f}")
print(f"{'='*50}")

metrics = {
    'task': 'prysm_phase',
    'task_number': 126,
    'method': 'Parametric phase diversity optimization (L-BFGS-B)',
    'phase_rmse_rad': round(float(phase_rmse_rad), 4),
    'phase_rmse_waves': round(float(phase_rmse_waves), 4),
    'phase_psnr_dB': round(float(phase_psnr), 2),
    'ssim': round(float(ssim_val), 4),
    'correlation_coefficient': round(float(cc), 6),
    'strehl_ratio_true': round(float(strehl_true), 6),
    'strehl_ratio_retrieved': round(float(strehl_retrieved), 6),
    'grid_size': NPIX,
    'wavelength_um': WAVELENGTH,
    'noise_photons': PHOTON_FLUX,
    'noise_readout': READOUT_NOISE,
    'defocus_diversity_waves': DEFOCUS_WAVES,
    'n_zernike_retrieval_modes': n_modes,
    'zernike_modes_truth': [{'n': n, 'm': m, 'coeff_waves': float(c)} for n, m, c in zernike_specs],
}

with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nSaved results/metrics.json")

# ============================================================
# 8. Save arrays
# ============================================================
np.save('results/ground_truth.npy', true_centered)
np.save('results/reconstruction.npy', retrieved_centered)
print("Saved results/ground_truth.npy, results/reconstruction.npy")

# ============================================================
# 9. Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

phase_vmin = np.min(true_centered[pupil_bool])
phase_vmax = np.max(true_centered[pupil_bool])

# (0,0) Ground truth phase
ax = axes[0, 0]
disp = true_centered.copy()
disp[~pupil_bool] = np.nan
im = ax.imshow(disp, cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
ax.set_title('Ground Truth Phase (rad)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Phase (rad)')
ax.axis('off')

# (0,1) Measured in-focus PSF
ax = axes[0, 1]
psf_d = psf_infocus_noisy.copy()
psf_d[psf_d <= 0] = 1e-10
im = ax.imshow(np.log10(psf_d), cmap='inferno')
ax.set_title('Measured PSF (in-focus, log10)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='log10(I)')
ax.axis('off')

# (0,2) Measured defocused PSF
ax = axes[0, 2]
psf_d2 = psf_defocus_noisy.copy()
psf_d2[psf_d2 <= 0] = 1e-10
im = ax.imshow(np.log10(psf_d2), cmap='inferno')
ax.set_title('Measured PSF (defocused, log10)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='log10(I)')
ax.axis('off')

# (1,0) Retrieved phase
ax = axes[1, 0]
disp = retrieved_centered.copy()
disp[~pupil_bool] = np.nan
im = ax.imshow(disp, cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
ax.set_title('Retrieved Phase (rad)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Phase (rad)')
ax.axis('off')

# (1,1) Phase error
ax = axes[1, 1]
disp = error_map.copy()
disp[~pupil_bool] = np.nan
err_lim = max(abs(np.nanmin(disp)), abs(np.nanmax(disp)), 0.01)
im = ax.imshow(disp, cmap='RdBu_r', vmin=-err_lim, vmax=err_lim)
ax.set_title(f'Phase Error (RMSE={phase_rmse_rad:.4f} rad)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Error (rad)')
ax.axis('off')

# (1,2) Coefficient comparison
ax = axes[1, 2]
mode_labels = []
true_vals = []
retr_vals = []
for i, (n, m) in enumerate(retrieval_nms):
    true_c = truth_dict.get((n, m), 0.0)
    if abs(true_c) > 0 or abs(coefs_opt[i]) > 0.005:
        mode_labels.append(f"Z({n},{m:+d})")
        true_vals.append(true_c)
        retr_vals.append(coefs_opt[i])

x_pos = np.arange(len(mode_labels))
width = 0.35
ax.bar(x_pos - width/2, true_vals, width, label='True', color='steelblue', alpha=0.8)
ax.bar(x_pos + width/2, retr_vals, width, label='Retrieved', color='coral', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Coefficient (waves)', fontsize=11)
ax.set_title('Zernike Coefficients', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Phase Retrieval: Parametric Diversity Optimization (prysm)\n'
             f'PSNR={phase_psnr:.2f} dB | SSIM={ssim_val:.4f} | CC={cc:.4f} | '
             f'RMSE={phase_rmse_waves:.4f} waves',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('results/reconstruction_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/reconstruction_result.png")

print("\nDone! Task 126 (prysm_phase) completed successfully.")
