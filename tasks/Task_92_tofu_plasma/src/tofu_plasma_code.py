#!/usr/bin/env python3
"""
Plasma/Fusion Tomography — Reconstruct 2D tokamak plasma emissivity from
line-integrated detector measurements (bolometry / SXR).

Forward model: y_i = integral_{LOS_i} epsilon(R, Z) dl   (line integrals)
Inverse:       Tikhonov-regularised least squares reconstruction.

This script is self-contained and uses only numpy/scipy/matplotlib/skimage.
"""

import matplotlib
matplotlib.use("Agg")

import json
import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Grid parameters (tokamak cross-section in R-Z plane)
NR, NZ = 40, 40           # reconstruction grid resolution
R_MIN, R_MAX = 1.0, 2.5  # major radius range [m]
Z_MIN, Z_MAX = -0.8, 0.8 # vertical range [m]

# Detector / LOS parameters
N_DETECTORS = 12          # number of detector fans
N_LOS_PER_DET = 80        # lines-of-sight per detector fan
NOISE_LEVEL = 0.005       # relative Gaussian noise on measurements

# Reconstruction
TIKHONOV_LAMBDA = 2e-3    # regularisation weight
LSQR_ITER_LIMIT = 3000

# Reproducibility
RNG_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# 1. Data generation — phantom plasma emissivity
# ──────────────────────────────────────────────────────────────────────────────

def make_grid():
    """Return 1-D R, Z arrays and the 2-D meshgrid."""
    r = np.linspace(R_MIN, R_MAX, NR)
    z = np.linspace(Z_MIN, Z_MAX, NZ)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")  # shape (NR, NZ)
    return r, z, RR, ZZ


def make_phantom(RR, ZZ):
    """
    Create a realistic tokamak-like emission phantom.
    Peaked profile centred at (R0, Z0) with an elliptical Gaussian shape
    plus a secondary weaker blob to add asymmetry.
    """
    R0, Z0 = 1.75, 0.0       # magnetic axis
    sigma_r, sigma_z = 0.30, 0.35

    # Main peaked profile
    eps = np.exp(-((RR - R0) ** 2 / (2 * sigma_r ** 2)
                   + (ZZ - Z0) ** 2 / (2 * sigma_z ** 2)))

    # Secondary blob (HFS accumulation)
    R1, Z1 = 1.45, 0.15
    sig1_r, sig1_z = 0.12, 0.10
    eps += 0.35 * np.exp(-((RR - R1) ** 2 / (2 * sig1_r ** 2)
                           + (ZZ - Z1) ** 2 / (2 * sig1_z ** 2)))

    # Clip outside last closed flux surface (rough ellipse)
    a_r, a_z = 0.60, 0.70
    mask = ((RR - R0) / a_r) ** 2 + ((ZZ - Z0) / a_z) ** 2 <= 1.0
    eps *= mask.astype(float)

    # Normalise to [0, 1]
    eps /= eps.max()
    return eps

# ──────────────────────────────────────────────────────────────────────────────
# 2. Forward operator — geometry matrix (line integrals)
# ──────────────────────────────────────────────────────────────────────────────

def _line_pixel_lengths(r_arr, z_arr, p0, p1, dr, dz):
    """
    Compute the intersection lengths of the line segment p0→p1 with each
    pixel on the (r_arr, z_arr) grid using Siddon's algorithm (simplified).

    Returns (row_indices, col_indices, values) for one LOS.
    """
    nr, nz = len(r_arr), len(z_arr)
    r0, z0 = p0
    r1, z1 = p1

    # Total line length
    total_len = np.hypot(r1 - r0, z1 - z0)
    if total_len < 1e-12:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Parametric: P(t) = p0 + t*(p1-p0), t in [0,1]
    n_samples = max(int(total_len / (min(dr, dz) * 0.25)), 500)
    t = np.linspace(0, 1, n_samples)
    r_pts = r0 + t * (r1 - r0)
    z_pts = z0 + t * (z1 - z0)

    # Map to grid indices
    ir = np.floor((r_pts - r_arr[0]) / dr).astype(int)
    iz = np.floor((z_pts - z_arr[0]) / dz).astype(int)

    # Mask valid
    valid = (ir >= 0) & (ir < nr) & (iz >= 0) & (iz < nz)
    ir = ir[valid]
    iz = iz[valid]

    if len(ir) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    flat = ir * nz + iz
    dl = total_len / n_samples  # approximate step length

    # Accumulate
    unique_idx, inverse = np.unique(flat, return_inverse=True)
    weights = np.bincount(inverse).astype(float) * dl

    return unique_idx, weights


def build_geometry_matrix(r_arr, z_arr):
    """
    Build the sparse geometry (line-integral) matrix L of shape
    (n_los_total, NR*NZ).

    Detector fans are placed around the vessel at different poloidal angles.
    """
    dr = r_arr[1] - r_arr[0]
    dz = z_arr[1] - z_arr[0]
    nr, nz = len(r_arr), len(z_arr)
    n_pix = nr * nz

    # Vessel centre for detector placement
    R_center = 0.5 * (R_MIN + R_MAX)
    Z_center = 0.5 * (Z_MIN + Z_MAX)
    vessel_radius = 1.0  # approximate distance from centre to wall

    # Detector angular positions (poloidal angle around vessel cross-section)
    det_angles = np.linspace(0, 2 * np.pi, N_DETECTORS, endpoint=False)

    rows, cols, vals = [], [], []
    los_idx = 0

    for da in det_angles:
        # Detector position on vessel wall
        det_r = R_center + vessel_radius * np.cos(da)
        det_z = Z_center + vessel_radius * np.sin(da)

        # Fan of LOS aiming through the plasma
        # Compute angular span that covers the plasma region
        fan_half = np.deg2rad(35)
        fan_angles = np.linspace(da + np.pi - fan_half,
                                 da + np.pi + fan_half,
                                 N_LOS_PER_DET)

        for fa in fan_angles:
            # End-point on opposite side of vessel
            end_r = det_r + 2.5 * vessel_radius * np.cos(fa)
            end_z = det_z + 2.5 * vessel_radius * np.sin(fa)

            idx, wts = _line_pixel_lengths(
                r_arr, z_arr, (det_r, det_z), (end_r, end_z), dr, dz
            )
            if len(idx) > 0:
                rows.extend([los_idx] * len(idx))
                cols.extend(idx.tolist())
                vals.extend(wts.tolist())
            los_idx += 1

    n_los = los_idx
    L = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_los, n_pix), dtype=np.float64
    )
    return L, n_los

# ──────────────────────────────────────────────────────────────────────────────
# 3. Inverse solver — Tikhonov regularisation via LSQR
# ──────────────────────────────────────────────────────────────────────────────

def _build_laplacian_2d(nr, nz):
    """Build a sparse 2D Laplacian operator for an (nr, nz) grid."""
    n = nr * nz
    diags = []
    offsets = []

    # Main diagonal: -4 (or fewer at boundaries handled by adjacency)
    main = -4.0 * np.ones(n)
    diags.append(main)
    offsets.append(0)

    # Right neighbour (+1 in z direction)
    d = np.ones(n - 1)
    # Zero out wrap-around at z boundaries
    for i in range(n - 1):
        if (i + 1) % nz == 0:
            d[i] = 0.0
    diags.append(d)
    offsets.append(1)

    # Left neighbour (-1 in z direction)
    d = np.ones(n - 1)
    for i in range(n - 1):
        if (i + 1) % nz == 0:
            d[i] = 0.0
    diags.append(d)
    offsets.append(-1)

    # Down neighbour (+nz in r direction)
    diags.append(np.ones(n - nz))
    offsets.append(nz)

    # Up neighbour (-nz in r direction)
    diags.append(np.ones(n - nz))
    offsets.append(-nz)

    Lap = sparse.diags(diags, offsets, shape=(n, n), format="csr")
    return Lap


def tikhonov_lsqr(L, y, lam, n_pix, max_iter=LSQR_ITER_LIMIT):
    """
    Solve  min_x || L x - y ||^2 + lam * || D x ||^2
    where D is the 2D Laplacian (smoothness prior).
    Stacked system:  [L; sqrt(lam)*D] x = [y; 0]
    """
    D = _build_laplacian_2d(NR, NZ)
    # Stack: [L; sqrt(lam)*D]
    A = sparse.vstack([L, np.sqrt(lam) * D], format="csr")
    b = np.concatenate([y, np.zeros(D.shape[0])])

    result = lsqr(A, b, iter_lim=max_iter, atol=1e-12, btol=1e-12)
    x_hat = result[0]
    # Enforce non-negativity (emissivity ≥ 0)
    x_hat = np.clip(x_hat, 0, None)
    return x_hat

# ──────────────────────────────────────────────────────────────────────────────
# 4. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(gt, recon):
    """Compute PSNR, SSIM, RMSE between ground truth and reconstruction."""
    data_range = gt.max() - gt.min()
    if data_range == 0:
        data_range = 1.0
    psnr = peak_signal_noise_ratio(gt, recon, data_range=data_range)
    ssim = structural_similarity(gt, recon, data_range=data_range)
    rmse = np.sqrt(np.mean((gt - recon) ** 2))
    return {"PSNR": round(float(psnr), 4),
            "SSIM": round(float(ssim), 4),
            "RMSE": round(float(rmse), 6)}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualise(r_arr, z_arr, gt, sinogram, recon, error, metrics, save_path):
    """4-panel figure: GT, sinogram, reconstruction, error map."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent_rz = [r_arr[0], r_arr[-1], z_arr[0], z_arr[-1]]

    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(gt.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title("(a) Ground Truth Emissivity ε(R,Z)")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (b) Line-integrated measurements (sinogram-like)
    ax = axes[0, 1]
    n_det = N_DETECTORS
    n_los = N_LOS_PER_DET
    sino_2d = sinogram.reshape(n_det, n_los) if len(sinogram) == n_det * n_los \
        else sinogram.reshape(-1, n_los)[:n_det]
    im = ax.imshow(sino_2d, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("(b) Line-Integrated Measurements")
    ax.set_xlabel("LOS index"); ax.set_ylabel("Detector fan")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (c) Reconstruction
    ax = axes[1, 0]
    im = ax.imshow(recon.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title(f"(c) Reconstruction  PSNR={metrics['PSNR']:.1f} dB  "
                 f"SSIM={metrics['SSIM']:.3f}")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (d) Error map
    ax = axes[1, 1]
    im = ax.imshow(error.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="seismic",
                   vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    ax.set_title(f"(d) Error Map  RMSE={metrics['RMSE']:.4f}")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Plasma/Fusion Tomography — Tokamak Emissivity Reconstruction",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualisation → {save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RNG_SEED)

    # --- Grid & phantom ---
    print("[1/6] Building grid and phantom …")
    r_arr, z_arr, RR, ZZ = make_grid()
    gt_2d = make_phantom(RR, ZZ)
    gt_flat = gt_2d.ravel()

    # --- Geometry matrix ---
    print("[2/6] Building geometry matrix (line-of-sight integrals) …")
    L, n_los = build_geometry_matrix(r_arr, z_arr)
    print(f"       Geometry matrix: {L.shape[0]} LOS × {L.shape[1]} pixels, "
          f"nnz = {L.nnz}")

    # --- Forward projection ---
    print("[3/6] Forward projection + noise …")
    y_clean = L @ gt_flat
    sigma_noise = NOISE_LEVEL * np.max(np.abs(y_clean))
    noise = rng.normal(0, sigma_noise, size=y_clean.shape)
    y_noisy = y_clean + noise
    print(f"       SNR ≈ {np.linalg.norm(y_clean) / np.linalg.norm(noise):.1f}")

    # --- Inverse reconstruction ---
    print("[4/6] Tikhonov-regularised LSQR reconstruction …")
    x_hat = tikhonov_lsqr(L, y_noisy, TIKHONOV_LAMBDA, L.shape[1])
    recon_2d = x_hat.reshape(NR, NZ)

    # Normalise reconstruction to same scale as GT for fair comparison
    if recon_2d.max() > 0:
        recon_2d = recon_2d / recon_2d.max() * gt_2d.max()

    # --- Metrics ---
    print("[5/6] Computing metrics …")
    metrics = compute_metrics(gt_2d, recon_2d)
    print(f"       PSNR = {metrics['PSNR']:.2f} dB")
    print(f"       SSIM = {metrics['SSIM']:.4f}")
    print(f"       RMSE = {metrics['RMSE']:.6f}")

    # --- Save artefacts ---
    print("[6/6] Saving results …")
    error_map = recon_2d - gt_2d

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_2d)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_2d)
    np.save(os.path.join(RESULTS_DIR, "measurements.npy"), y_noisy)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"       Metrics → {os.path.join(RESULTS_DIR, 'metrics.json')}")

    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualise(r_arr, z_arr, gt_2d, y_noisy, recon_2d, error_map, metrics, vis_path)

    print("\n✓ Pipeline complete.")
    return metrics


if __name__ == "__main__":
    main()
