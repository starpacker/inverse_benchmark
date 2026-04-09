#!/usr/bin/env python
"""
Task 183: priism_radio
Sparse modeling radio interferometric imaging: reconstruct radio images
from visibility data using L1/TSV regularization (ISTA solver).

Implements the core algorithm from the priism package without CASA dependency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import json
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter


# ============================================================
# 1. Sky Model Synthesis
# ============================================================

def create_sky_model(nx=128, ny=128, rng=None):
    """Create a synthetic sky model with point sources and extended emission."""
    if rng is None:
        rng = np.random.default_rng(42)

    sky = np.zeros((ny, nx), dtype=np.float64)

    # 3 point sources at different positions/fluxes
    point_sources = [
        (64, 64, 1.0),    # center, bright
        (40, 80, 0.5),    # offset, medium
        (90, 45, 0.3),    # offset, dim
    ]
    for y, x, flux in point_sources:
        sky[y, x] = flux

    # 1 extended Gaussian source (simulating a galaxy)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = 75, 55
    sigma_x, sigma_y = 5.0, 3.5
    angle = np.pi / 6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = xx - cx
    dy = yy - cy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    gauss = 0.4 * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    sky += gauss

    return sky


# ============================================================
# 2. UV Coverage Generation (Earth Rotation Synthesis)
# ============================================================

def generate_uv_coverage(n_antennas=10, n_hours=6, n_time_steps=60, rng=None):
    """
    Simulate (u,v) coverage from an interferometric array via
    earth-rotation synthesis.

    Returns:
        u, v: 1-D arrays of (u,v) coordinates in units of pixels
              (matched to image grid spacing).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Random antenna positions on ground (East, North) in units proportional
    # to max baseline ~ image size / 2
    max_baseline = 50.0  # in pixel units
    ant_E = rng.uniform(-max_baseline, max_baseline, n_antennas)
    ant_N = rng.uniform(-max_baseline, max_baseline, n_antennas)

    # Declination of source (radians)
    dec = np.deg2rad(45.0)

    # Hour angles spanning the observation
    ha = np.linspace(-n_hours / 2, n_hours / 2, n_time_steps) * (np.pi / 12)

    u_all, v_all = [], []
    for i in range(n_antennas):
        for j in range(i + 1, n_antennas):
            bE = ant_E[j] - ant_E[i]
            bN = ant_N[j] - ant_N[i]
            # Projected baselines
            u_t = bE * np.cos(ha) - bN * np.sin(ha) * np.sin(dec)
            v_t = bE * np.sin(ha) * np.sin(dec) + bN * np.cos(ha) * np.cos(dec)  # simplified
            u_all.append(u_t)
            v_all.append(v_t)

    u = np.concatenate(u_all)
    v = np.concatenate(v_all)

    # Also include conjugate baselines (Hermitian symmetry)
    u = np.concatenate([u, -u])
    v = np.concatenate([v, -v])

    return u, v


# ============================================================
# 3. Forward Operator (Sparse Fourier Sampling)
# ============================================================

def _uv_to_grid_indices(u, v, nx, ny):
    """Convert continuous (u,v) to nearest grid indices for FFT grid."""
    # Shift to [0, nx) range
    ui = np.round(u).astype(int) % nx
    vi = np.round(v).astype(int) % ny
    return ui, vi


def forward_operator(image, ui, vi):
    """
    Forward model: image → visibilities at (u,v) sample points.
    Uses FFT + sampling.
    """
    ft = np.fft.fft2(image)
    vis = ft[vi, ui]
    return vis


def adjoint_operator(vis, ui, vi, nx, ny):
    """
    Adjoint model: visibilities → image (dirty image direction).
    Places visibilities on grid and applies inverse FFT.
    """
    grid = np.zeros((ny, nx), dtype=complex)
    # Accumulate visibilities on grid
    np.add.at(grid, (vi, ui), vis)
    img = np.fft.ifft2(grid).real
    return img


def make_dirty_image(vis, ui, vi, nx, ny):
    """Create the dirty image (adjoint applied to visibilities), normalized."""
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    # Also make PSF for normalization
    psf_grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(psf_grid, (vi, ui), 1.0)
    dirty = np.fft.ifft2(grid).real
    psf = np.fft.ifft2(psf_grid).real
    peak_psf = psf.max()
    if peak_psf > 0:
        dirty /= peak_psf
    return dirty


# ============================================================
# 4. TSV Regularization
# ============================================================

def tsv_value(image):
    """Total Squared Variation: sum of squared differences."""
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    return np.sum(dx ** 2) + np.sum(dy ** 2)


def tsv_gradient(image):
    """Gradient of TSV(I) w.r.t. I."""
    ny, nx = image.shape
    grad = np.zeros_like(image)
    # d/dI[i,j] of (I[i,j+1]-I[i,j])^2 terms
    # For horizontal differences
    grad[:, :-1] -= 2 * (image[:, 1:] - image[:, :-1])
    grad[:, 1:] += 2 * (image[:, 1:] - image[:, :-1])
    # For vertical differences
    grad[:-1, :] -= 2 * (image[1:, :] - image[:-1, :])
    grad[1:, :] += 2 * (image[1:, :] - image[:-1, :])
    return grad


# ============================================================
# 5. ISTA Solver with L1 + TSV
# ============================================================

def soft_threshold(x, threshold):
    """Proximal operator for L1 norm."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def ista_l1_tsv(vis, ui, vi, nx, ny, lambda_l1=1e-4, lambda_tsv=5e-4,
                step_size=None, max_iter=500, verbose=True):
    """
    ISTA solver for:
        min_{I>=0} ||Phi*I - V||^2 + lambda_l1*||I||_1 + lambda_tsv*TSV(I)

    Parameters
    ----------
    vis : complex array — measured visibilities
    ui, vi : int arrays — grid indices of (u,v) samples
    nx, ny : int — image dimensions
    lambda_l1 : float — L1 sparsity weight
    lambda_tsv : float — TSV smoothness weight
    step_size : float or None — if None, estimate from operator norm
    max_iter : int — maximum iterations

    Returns
    -------
    image : 2D array — reconstructed image
    history : dict — convergence history
    """
    n_vis = len(vis)

    # Estimate step size from Lipschitz constant of data fidelity gradient.
    # L = ||Phi^H Phi|| ~ n_vis (since each visibility is an FFT sample).
    # Use a safe estimate.
    if step_size is None:
        # Power iteration to estimate operator norm (a few iterations)
        x = np.random.randn(ny, nx)
        for _ in range(20):
            y = adjoint_operator(forward_operator(x, ui, vi), ui, vi, nx, ny)
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-14:
                break
            x = y / norm_y
        L = norm_y
        step_size = 0.9 / L
        if verbose:
            print(f"Estimated Lipschitz constant L={L:.2e}, step_size={step_size:.2e}")

    # Initialize with dirty image (good starting point)
    image = make_dirty_image(vis, ui, vi, nx, ny)
    image = np.maximum(image, 0.0)

    history = {'cost': [], 'data_fidelity': [], 'l1': [], 'tsv': []}

    for it in range(max_iter):
        # Gradient of data fidelity: Phi^H (Phi*I - V)
        residual = forward_operator(image, ui, vi) - vis
        grad_data = adjoint_operator(residual, ui, vi, nx, ny)

        # TSV gradient
        grad_tsv = tsv_gradient(image)

        # Gradient descent step
        image = image - step_size * (grad_data + lambda_tsv * grad_tsv)

        # L1 proximal (soft thresholding)
        image = soft_threshold(image, step_size * lambda_l1)

        # Non-negativity constraint
        image = np.maximum(image, 0.0)

        # Track convergence (every 50 iterations)
        if it % 50 == 0 or it == max_iter - 1:
            df = 0.5 * np.sum(np.abs(residual) ** 2)
            l1_val = lambda_l1 * np.sum(np.abs(image))
            tsv_val = lambda_tsv * tsv_value(image)
            cost = df + l1_val + tsv_val
            history['cost'].append(cost)
            history['data_fidelity'].append(df)
            history['l1'].append(l1_val)
            history['tsv'].append(tsv_val)
            if verbose and it % 100 == 0:
                print(f"  iter {it:4d}: cost={cost:.4e}  "
                      f"data={df:.4e}  L1={l1_val:.4e}  TSV={tsv_val:.4e}")

    return image, history


# ============================================================
# 6. Metrics
# ============================================================

def compute_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(gt, recon):
    """Structural similarity index."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)


def compute_cc(gt, recon):
    """Pearson correlation coefficient."""
    g = gt.ravel()
    r = recon.ravel()
    g_m = g - g.mean()
    r_m = r - r.mean()
    num = np.sum(g_m * r_m)
    den = np.sqrt(np.sum(g_m ** 2) * np.sum(r_m ** 2))
    if den < 1e-15:
        return 0.0
    return float(num / den)


def compute_dynamic_range(image, source_mask):
    """Ratio of peak signal to rms in background region."""
    bg = image[~source_mask]
    rms = np.sqrt(np.mean(bg ** 2)) if len(bg) > 0 else 1e-15
    if rms < 1e-15:
        rms = 1e-15
    return float(image.max() / rms)


# ============================================================
# 7. Visualization
# ============================================================

def make_figure(sky_gt, dirty, recon, u, v, save_path):
    """Create 5-panel figure: GT, dirty, recon, error, uv-coverage."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    vmin_log = max(sky_gt[sky_gt > 0].min() * 0.1, 1e-4) if np.any(sky_gt > 0) else 1e-4
    vmax = sky_gt.max()

    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(sky_gt, origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=vmin_log, vmax=vmax))
    ax.set_title('(a) Ground Truth Sky', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    # (b) Dirty image
    ax = axes[0, 1]
    im = ax.imshow(dirty, origin='lower', cmap='inferno')
    ax.set_title('(b) Dirty Image', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    # (c) Sparse reconstruction
    ax = axes[0, 2]
    im = ax.imshow(recon, origin='lower', cmap='inferno',
                   vmin=0, vmax=vmax)
    ax.set_title('(c) L1+TSV Reconstruction', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    # (d) Error map
    ax = axes[1, 0]
    error = np.abs(sky_gt - recon)
    im = ax.imshow(error, origin='lower', cmap='hot')
    ax.set_title('(d) Error |GT - Recon|', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='|Error|')

    # (e) UV coverage
    ax = axes[1, 1]
    ax.scatter(u, v, s=0.3, alpha=0.3, color='cyan', edgecolors='none')
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlabel('u (pixels)', fontsize=11)
    ax.set_ylabel('v (pixels)', fontsize=11)
    ax.set_title('(e) (u,v) Coverage', fontsize=13, fontweight='bold')

    # (f) Convergence or residual
    ax = axes[1, 2]
    # Show cross-sections through brightest source
    cy, cx = 64, 64
    ax.plot(sky_gt[cy, :], 'b-', linewidth=1.5, label='GT row')
    ax.plot(recon[cy, :], 'r--', linewidth=1.5, label='Recon row')
    ax.plot(sky_gt[:, cx], 'b:', linewidth=1.5, label='GT col')
    ax.plot(recon[:, cx], 'r:', linewidth=1.5, label='Recon col')
    ax.set_xlabel('Pixel', fontsize=11)
    ax.set_ylabel('Flux', fontsize=11)
    ax.set_title('(f) Cross-section (row/col through center)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('Task 183: Radio Interferometric Imaging (L1+TSV Sparse Reconstruction)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {save_path}")


# ============================================================
# 8. Main Pipeline
# ============================================================

def main():
    rng = np.random.default_rng(42)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    nx, ny = 128, 128

    # --- 1. Create sky model ---
    print("=" * 60)
    print("Step 1: Creating sky model ...")
    sky_gt = create_sky_model(nx, ny, rng=rng)
    print(f"  Sky shape: {sky_gt.shape}, max flux: {sky_gt.max():.4f}")

    # --- 2. Generate UV coverage ---
    print("Step 2: Generating (u,v) coverage ...")
    u, v = generate_uv_coverage(n_antennas=10, n_hours=6, n_time_steps=60, rng=rng)
    print(f"  Number of (u,v) points (incl. conjugates): {len(u)}")

    # Map to grid indices
    ui, vi = _uv_to_grid_indices(u, v, nx, ny)

    # Remove duplicate grid points for cleaner sampling
    uv_pairs = np.stack([ui, vi], axis=1)
    _, unique_idx = np.unique(uv_pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    ui = ui[unique_idx]
    vi = vi[unique_idx]
    u_unique = u[unique_idx]
    v_unique = v[unique_idx]
    print(f"  Unique grid points: {len(ui)}")

    # --- 3. Compute visibilities ---
    print("Step 3: Computing visibilities ...")
    vis_true = forward_operator(sky_gt, ui, vi)
    # Add noise (SNR ~30)
    signal_power = np.mean(np.abs(vis_true) ** 2)
    noise_std = np.sqrt(signal_power / 30.0)
    noise = noise_std * (rng.standard_normal(len(vis_true)) +
                         1j * rng.standard_normal(len(vis_true))) / np.sqrt(2)
    vis_noisy = vis_true + noise
    actual_snr = np.sqrt(np.mean(np.abs(vis_true) ** 2) / np.mean(np.abs(noise) ** 2))
    print(f"  Noise std: {noise_std:.4e}, actual SNR: {actual_snr:.1f}")

    # --- 4. Dirty image ---
    print("Step 4: Making dirty image ...")
    dirty = make_dirty_image(vis_noisy, ui, vi, nx, ny)
    print(f"  Dirty image range: [{dirty.min():.4f}, {dirty.max():.4f}]")

    # --- 5. ISTA reconstruction ---
    print("Step 5: Running ISTA (L1+TSV) reconstruction ...")
    recon, history = ista_l1_tsv(
        vis_noisy, ui, vi, nx, ny,
        lambda_l1=2e-4,
        lambda_tsv=1e-3,
        max_iter=800,
        verbose=True,
    )
    print(f"  Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")

    # --- 6. Evaluate ---
    print("Step 6: Computing metrics ...")
    psnr_val = compute_psnr(sky_gt, recon)
    ssim_val = compute_ssim(sky_gt, recon)
    cc_val = compute_cc(sky_gt, recon)

    # Source mask for dynamic range (pixels > 5% of max)
    source_mask = sky_gt > 0.05 * sky_gt.max()
    dr_val = compute_dynamic_range(recon, source_mask)

    # Also compute dirty image metrics for comparison
    dirty_norm = np.maximum(dirty, 0)
    dirty_norm = dirty_norm / dirty_norm.max() * sky_gt.max() if dirty_norm.max() > 0 else dirty_norm
    psnr_dirty = compute_psnr(sky_gt, dirty_norm)

    print(f"  PSNR (reconstruction): {psnr_val:.2f} dB")
    print(f"  PSNR (dirty image):    {psnr_dirty:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  CC:   {cc_val:.4f}")
    print(f"  Dynamic Range: {dr_val:.1f}")

    metrics = {
        'task_id': 183,
        'task_name': 'priism_radio',
        'method': 'ISTA with L1+TSV regularization',
        'PSNR_dB': round(psnr_val, 2),
        'SSIM': round(ssim_val, 4),
        'CC': round(cc_val, 4),
        'dynamic_range': round(dr_val, 1),
        'PSNR_dirty_dB': round(psnr_dirty, 2),
        'n_uv_points': int(len(ui)),
        'image_size': [nx, ny],
        'lambda_l1': 2e-4,
        'lambda_tsv': 1e-3,
        'max_iter': 800,
    }

    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # --- 7. Visualization ---
    print("Step 7: Creating visualization ...")
    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    make_figure(sky_gt, dirty, recon, u_unique, v_unique, fig_path)

    # --- 8. Save arrays ---
    print("Step 8: Saving arrays ...")
    np.save(os.path.join(results_dir, 'ground_truth.npy'), sky_gt)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon)
    np.save(os.path.join(results_dir, 'dirty_image.npy'), dirty)
    np.save(os.path.join(results_dir, 'visibilities.npy'), vis_noisy)
    np.save(os.path.join(results_dir, 'uv_coords.npy'), np.stack([u_unique, v_unique]))

    print("=" * 60)
    print("DONE. All outputs saved to results/")
    print(f"  PSNR = {psnr_val:.2f} dB  (target > 20 dB)")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  CC   = {cc_val:.4f}")

    # Validation
    assert psnr_val > 20.0, f"PSNR {psnr_val:.2f} dB < 20 dB target!"
    print("✓ PSNR > 20 dB — PASS")


if __name__ == '__main__':
    main()
