"""
pyTFM - Traction Force Microscopy Inverse Problem
===================================================
Task: Recover cell traction forces from gel surface displacement fields
      using Fourier Transform Traction Cytometry (FTTC).

Inverse Problem:
    Given measured displacement field (u, v) on an elastic substrate surface,
    recover the traction force field (tx, ty) that caused those displacements.

Forward Model (Boussinesq):
    (u, v) = G * (tx, ty)   [convolution via Fourier-space Green's function]
    where G is the Boussinesq elastic Green's function for a half-space.

Inverse Solver:
    (tx, ty) = G^{-1} * (u, v)  [Fourier-space inversion with spatial filtering]

Repo: https://github.com/fabrylab/pyTFM
Paper: Butler et al. (2002), Traction fields, moments, and strain energy that
       cells exert on their surroundings. Am J Physiol Cell Physiol.

Usage:
    /data/yjh/pyTFM_env/bin/python pyTFM_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

# Add repo to path
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from pyTFM.TFM_functions import ffttc_traction
import scipy.fft

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physical parameters
YOUNG_MODULUS = 49000.0       # Pa (Young's modulus of gel substrate)
POISSON_RATIO = 0.49          # Poisson's ratio (nearly incompressible)
# In a real TFM experiment, pixelsize1 = microscope image pixel size,
# pixelsize2 = deformation field pixel size (typically pixelsize1 * (window - overlap))
# For our synthetic benchmark, we work directly on the traction/deformation grid,
# so pixelsize1 = pixelsize2 = physical pixel size of that grid.
PIXEL_SIZE = 6.44e-6          # m/pixel (~6.44 µm, typical for 0.201 µm/px image with 32-px PIV step)
NOISE_STD = 0.02              # noise standard deviation in pixels (displacement noise)
GRID_SIZE = 64                # size of synthetic traction field (pixels)

# ═══════════════════════════════════════════════════════════
# 2. Forward Operator (Traction → Displacement)
# ═══════════════════════════════════════════════════════════
def forward_operator(tx, ty, pixelsize, young, sigma=0.49):
    """
    Boussinesq forward model (infinite half-space): compute surface displacements
    from traction forces using Fourier-space Green's function.

    CRITICAL: The padding and wave vector conventions MUST exactly match
    ffttc_traction in pyTFM/TFM_functions.py for perfect round-trip consistency.

    The inverse solver computes:  T = K_inv * FFT(u * pixelsize1)
    So the forward is:  u = IFFT(K * T_ft) / pixelsize1

    where K_inv is the elastic stiffness kernel and K = K_inv^{-1} is the
    compliance (Green's function) kernel.

    Parameters:
        tx, ty: 2D arrays, traction fields in Pa
        pixelsize: float, pixel size in meters (same for pixelsize1 and pixelsize2)
        young: float, Young's modulus in Pa
        sigma: float, Poisson's ratio

    Returns:
        u, v: 2D arrays, displacement fields in PIXELS
    """
    ax1_length, ax2_length = tx.shape
    max_ind = int(max(ax1_length, ax2_length))
    if max_ind % 2 != 0:
        max_ind += 1

    # MUST match inverse solver's padding convention: top-left placement
    tx_expand = np.zeros((max_ind, max_ind))
    ty_expand = np.zeros((max_ind, max_ind))
    tx_expand[:ax1_length, :ax2_length] = tx
    ty_expand[:ax1_length, :ax2_length] = ty

    # Wave vectors — MUST match ffttc_traction exactly
    # In ffttc_traction: kx = ... * 2*pi, then k = sqrt(kx^2 + ky^2) / (pixelsize2 * max_ind)
    kx1 = np.array([list(range(0, int(max_ind / 2), 1)), ] * int(max_ind))
    kx2 = np.array([list(range(-int(max_ind / 2), 0, 1)), ] * int(max_ind))
    kx = np.append(kx1, kx2, axis=1) * 2 * np.pi
    ky = np.transpose(kx)
    k = np.sqrt(kx ** 2 + ky ** 2) / (pixelsize * max_ind)

    # Angle (same as inverse solver)
    alpha = np.arctan2(ky, kx)
    alpha[0, 0] = np.pi / 2

    # The inverse solver's K_inv kernel:
    # kix = (k*E)/(2*(1-sigma^2)) * (1-sigma + sigma*cos(alpha)^2)
    # kiy = (k*E)/(2*(1-sigma^2)) * (1-sigma + sigma*sin(alpha)^2)
    # kid = (k*E)/(2*(1-sigma^2)) * sigma*sin(alpha)*cos(alpha)
    #
    # Forward kernel G = K_inv^{-1}: we need to invert the 2x2 system at each k:
    # [kix kid] [u_ft]   [tx_ft]
    # [kid kiy] [v_ft] = [ty_ft]  (after scaling u by pixelsize1)
    #
    # So: [u_ft] = 1/det * [ kiy -kid] [tx_ft]
    #     [v_ft]            [-kid  kix] [ty_ft]
    # where det = kix*kiy - kid^2

    kix = ((k * young) / (2 * (1 - sigma ** 2))) * (1 - sigma + sigma * np.cos(alpha) ** 2)
    kiy = ((k * young) / (2 * (1 - sigma ** 2))) * (1 - sigma + sigma * np.sin(alpha) ** 2)
    kid = ((k * young) / (2 * (1 - sigma ** 2))) * (sigma * np.sin(alpha) * np.cos(alpha))

    # Zero out cross terms at Nyquist (same as inverse solver)
    kid[:, int(max_ind / 2)] = np.zeros(max_ind)
    kid[int(max_ind / 2), :] = np.zeros(max_ind)

    # Determinant of the K_inv 2x2 matrix
    det = kix * kiy - kid ** 2

    # Avoid division by zero at DC
    det[0, 0] = 1.0  # will be zeroed out anyway

    # Green's function (inverse of K_inv)
    g11 = kiy / det
    g12 = -kid / det
    g22 = kix / det

    # FFT of traction fields
    tx_ft = scipy.fft.fft2(tx_expand)
    ty_ft = scipy.fft.fft2(ty_expand)

    # Displacement in Fourier space (in meters, since inverse uses u*pixelsize1)
    u_ft = g11 * tx_ft + g12 * ty_ft
    v_ft = g12 * tx_ft + g22 * ty_ft

    # Zero DC component (mean displacement is unconstrained)
    u_ft[0, 0] = 0
    v_ft[0, 0] = 0

    # Back to real space
    u = scipy.fft.ifft2(u_ft).real
    v = scipy.fft.ifft2(v_ft).real

    # Cut to original size (MUST match inverse solver's placement)
    u_cut = u[:ax1_length, :ax2_length]
    v_cut = v[:ax1_length, :ax2_length]

    # The inverse solver multiplies u by pixelsize1 before FFT:
    #   u_ft_inv = FFT(u_pixels * pixelsize1)
    #   tx_ft = kix * u_ft_inv + kid * v_ft_inv
    #
    # So the forward result in meters is: u_meters = u_cut (from above)
    # And u_pixels = u_meters / pixelsize1
    return u_cut / pixelsize, v_cut / pixelsize


# ═══════════════════════════════════════════════════════════
# 3. Data Generation (Synthetic Benchmark)
# ═══════════════════════════════════════════════════════════
def generate_synthetic_traction_field(N=64):
    """
    Generate a synthetic cell-like traction force dipole pattern.
    Models a single cell contracting on the substrate — two opposing
    Gaussian-shaped force patches (contractile dipole).

    Returns:
        tx_gt, ty_gt: ground truth traction fields in Pa
    """
    y, x = np.mgrid[:N, :N]
    cx, cy = N // 2, N // 2

    # Two Gaussian blobs offset from center (contractile dipole)
    sigma_blob = N / 8.0
    offset = N / 5.0

    # Left blob (pulling right)
    g_left = np.exp(-((x - (cx - offset))**2 + (y - cy)**2) / (2 * sigma_blob**2))
    # Right blob (pulling left)
    g_right = np.exp(-((x - (cx + offset))**2 + (y - cy)**2) / (2 * sigma_blob**2))

    # Traction magnitudes (typical cell traction: 100-1000 Pa)
    max_traction = 500.0  # Pa

    # tx: left blob pulls right (+), right blob pulls left (-)
    tx_gt = max_traction * g_left - max_traction * g_right
    # ty: add some vertical component (realistic cell shape)
    ty_gt = 0.3 * max_traction * g_left * (y - cy) / (sigma_blob * 2)
    ty_gt -= 0.3 * max_traction * g_right * (y - cy) / (sigma_blob * 2)

    return tx_gt.astype(np.float64), ty_gt.astype(np.float64)


def load_or_generate_data():
    """
    Generate synthetic benchmark data:
    1. Create ground truth traction field
    2. Apply forward model to get displacement field (in pixels)
    3. Add measurement noise
    """
    print("[DATA] Generating synthetic traction force dipole...")
    tx_gt, ty_gt = generate_synthetic_traction_field(N=GRID_SIZE)

    # Forward model: traction → displacement (returns pixels)
    print("[DATA] Applying forward operator (Boussinesq Green's function)...")
    u_clean_px, v_clean_px = forward_operator(tx_gt, ty_gt, PIXEL_SIZE, YOUNG_MODULUS, POISSON_RATIO)

    print(f"[DATA] Max displacement: {np.max(np.sqrt(u_clean_px**2 + v_clean_px**2)):.4f} pixels")

    # Add Gaussian noise to displacement
    np.random.seed(42)
    u_noisy_px = u_clean_px + NOISE_STD * np.random.randn(*u_clean_px.shape)
    v_noisy_px = v_clean_px + NOISE_STD * np.random.randn(*v_clean_px.shape)

    metadata = {
        "young": YOUNG_MODULUS,
        "sigma": POISSON_RATIO,
        "pixelsize1": PIXEL_SIZE,     # used by ffttc_traction to scale u,v from pixels to meters
        "pixelsize2": PIXEL_SIZE,     # used by ffttc_traction for wave vector computation
        "noise_std": NOISE_STD,
    }

    return (u_noisy_px, v_noisy_px), (tx_gt, ty_gt), metadata


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver (Displacement → Traction)
# ═══════════════════════════════════════════════════════════
def reconstruct(measurements, **params):
    """
    FTTC inverse solver: recover traction forces from displacement field.

    Uses pyTFM's Fourier Transform Traction Cytometry with Gaussian filtering
    for regularization (noise suppression in high-frequency components).

    Parameters:
        measurements: tuple (u, v) displacement fields in pixels
        params: must include young, sigma, pixelsize1, pixelsize2

    Returns:
        tx_recon, ty_recon: reconstructed traction fields in Pa
    """
    u, v = measurements
    young = params["young"]
    sigma = params["sigma"]
    pixelsize1 = params["pixelsize1"]
    pixelsize2 = params["pixelsize2"]

    # Use pyTFM's FTTC solver
    # Note: spatial_filter=None for no filtering (best for low noise)
    # For noisy data, use spatial_filter="gaussian" with appropriate fs
    tx_recon, ty_recon = ffttc_traction(
        u, v,
        pixelsize1=pixelsize1,
        pixelsize2=pixelsize2,
        young=young,
        sigma=sigma,
        spatial_filter="gaussian",
        fs=3  # smaller filter for less smoothing (in auto-units)
    )

    return tx_recon, ty_recon


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_psnr(ref, test, data_range=None):
    """Compute PSNR (dB) between reference and test arrays."""
    if data_range is None:
        data_range = ref.max() - ref.min()
    if data_range == 0:
        return float('inf') if np.allclose(ref, test) else 0.0
    mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(ref, test):
    """Compute SSIM for 2D fields."""
    from skimage.metrics import structural_similarity as ssim
    data_range = ref.max() - ref.min()
    if data_range == 0:
        data_range = 1.0
    return ssim(ref, test, data_range=data_range)


def compute_rmse(ref, test):
    """Compute RMSE."""
    return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))


def compute_relative_error(ref, test):
    """Compute relative error (RE) = ||ref - test||_2 / ||ref||_2."""
    ref_norm = np.linalg.norm(ref.ravel())
    if ref_norm == 0:
        return float('inf')
    return np.linalg.norm((ref - test).ravel()) / ref_norm


def compute_correlation_coefficient(ref, test):
    """Compute Pearson correlation coefficient."""
    ref_flat = ref.ravel()
    test_flat = test.ravel()
    if np.std(ref_flat) == 0 or np.std(test_flat) == 0:
        return 0.0
    return float(np.corrcoef(ref_flat, test_flat)[0, 1])


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(tx_gt, ty_gt, u_meas, v_meas, tx_rec, ty_rec, metrics, save_path):
    """
    Generate comprehensive visualization:
    Row 1: Ground truth traction magnitude | Measured displacement magnitude |
           Reconstructed traction magnitude | Error map
    Row 2: Quiver plots of GT tractions | measured displacements |
           reconstructed tractions | error vectors
    """
    gt_mag = np.sqrt(tx_gt**2 + ty_gt**2)
    rec_mag = np.sqrt(tx_rec**2 + ty_rec**2)
    disp_mag = np.sqrt(u_meas**2 + v_meas**2)
    err_mag = np.abs(gt_mag - rec_mag)

    vmax = max(gt_mag.max(), rec_mag.max())

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 1: Scalar magnitude fields
    im0 = axes[0, 0].imshow(gt_mag, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title("Ground Truth |T| (Pa)", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(disp_mag, cmap='viridis')
    axes[0, 1].set_title("Measured |u| (pixels)", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(rec_mag, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 2].set_title("Reconstructed |T| (Pa)", fontsize=12)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[0, 3].imshow(err_mag, cmap='magma')
    axes[0, 3].set_title("Error |GT - Recon| (Pa)", fontsize=12)
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # Row 2: Quiver plots
    N = tx_gt.shape[0]
    skip = max(1, N // 16)  # subsample for readability
    y_grid, x_grid = np.mgrid[:N, :N]
    sl = (slice(None, None, skip), slice(None, None, skip))

    axes[1, 0].quiver(x_grid[sl], y_grid[sl], tx_gt[sl], ty_gt[sl],
                       gt_mag[sl], cmap='hot', scale_units='xy')
    axes[1, 0].set_title("GT Traction Vectors", fontsize=12)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()

    axes[1, 1].quiver(x_grid[sl], y_grid[sl], u_meas[sl], v_meas[sl],
                       disp_mag[sl], cmap='viridis', scale_units='xy')
    axes[1, 1].set_title("Measured Displacement Vectors", fontsize=12)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()

    axes[1, 2].quiver(x_grid[sl], y_grid[sl], tx_rec[sl], ty_rec[sl],
                       rec_mag[sl], cmap='hot', scale_units='xy')
    axes[1, 2].set_title("Reconstructed Traction Vectors", fontsize=12)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].invert_yaxis()

    err_tx = tx_gt - tx_rec
    err_ty = ty_gt - ty_rec
    axes[1, 3].quiver(x_grid[sl], y_grid[sl], err_tx[sl], err_ty[sl],
                       err_mag[sl], cmap='magma', scale_units='xy')
    axes[1, 3].set_title("Error Vectors", fontsize=12)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].invert_yaxis()

    fig.suptitle(
        f"pyTFM — Traction Force Microscopy Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"RMSE={metrics['rmse']:.2f} Pa | RE={metrics['relative_error']:.4f} | "
        f"CC={metrics['correlation_coefficient']:.4f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  pyTFM — Traction Force Microscopy Inverse Problem")
    print("=" * 60)

    # (a) Generate synthetic data
    measurements, ground_truth, metadata = load_or_generate_data()
    u_meas, v_meas = measurements
    tx_gt, ty_gt = ground_truth
    print(f"[DATA] Displacement field shape: {u_meas.shape}")
    print(f"[DATA] GT traction field shape: {tx_gt.shape}")

    # (b) Run inverse solver (FTTC)
    print("[RECON] Running FTTC inverse solver...")
    tx_rec, ty_rec = reconstruct(measurements, **metadata)
    print(f"[RECON] Reconstructed traction field shape: {tx_rec.shape}")

    # (c) Compute metrics on traction magnitude
    gt_mag = np.sqrt(tx_gt**2 + ty_gt**2)
    rec_mag = np.sqrt(tx_rec**2 + ty_rec**2)

    metrics = {
        "psnr": float(compute_psnr(gt_mag, rec_mag)),
        "ssim": float(compute_ssim(gt_mag, rec_mag)),
        "rmse": float(compute_rmse(gt_mag, rec_mag)),
        "relative_error": float(compute_relative_error(gt_mag, rec_mag)),
        "correlation_coefficient": float(compute_correlation_coefficient(gt_mag, rec_mag)),
        # Also compute component-wise metrics
        "psnr_tx": float(compute_psnr(tx_gt, tx_rec)),
        "psnr_ty": float(compute_psnr(ty_gt, ty_rec)),
        "rmse_tx": float(compute_rmse(tx_gt, tx_rec)),
        "rmse_ty": float(compute_rmse(ty_gt, ty_rec)),
    }

    print(f"[EVAL] PSNR (magnitude) = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] SSIM (magnitude) = {metrics['ssim']:.6f}")
    print(f"[EVAL] RMSE (magnitude) = {metrics['rmse']:.4f} Pa")
    print(f"[EVAL] Relative Error   = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Correlation Coef = {metrics['correlation_coefficient']:.6f}")

    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # (e) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(tx_gt, ty_gt, u_meas, v_meas, tx_rec, ty_rec, metrics, vis_path)

    # (f) Save arrays
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), np.stack([tx_rec, ty_rec]))
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), np.stack([tx_gt, ty_gt]))
    np.save(os.path.join(RESULTS_DIR, "measurements.npy"), np.stack([u_meas, v_meas]))

    print("=" * 60)
    print("  DONE")
    print("=" * 60)
