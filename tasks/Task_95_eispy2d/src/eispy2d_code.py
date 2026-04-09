"""
eispy2d - EM Inverse Scattering (Born Approximation)
=====================================================
Reconstruct 2D dielectric contrast profile from scattered EM field data.

Physics:
  - 2D TM-polarized scattering with plane-wave illumination
  - Born approximation linearizes E_scat = A @ chi
  - Tikhonov regularisation for the inverse solve
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.special import hankel2

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_95_eispy2d"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── physical / numerical parameters ───────────────────────────────
N_GRID      = 32        # pixels per side
DOMAIN_SIZE = 0.30      # 30 cm square investigation domain
FREQ        = 3.0e9     # 3 GHz  → wavelength = 10 cm
C0          = 3.0e8
K0          = 2.0 * np.pi * FREQ / C0
WAVELENGTH  = C0 / FREQ
N_TX        = 36        # transmitters equally spaced on circle
N_RX        = 36        # receivers equally spaced on circle
R_ARRAY     = 0.50      # array radius  (50 cm, well outside domain)
SNR_DB      = 30.0
LAMBDA_REG  = "auto"    # Tikhonov parameter — set via L-curve heuristic


# ====================================================================
# 1. Phantom
# ====================================================================
def create_phantom(N, L):
    """Two dielectric cylinders inside the investigation domain.

    Returns
    -------
    chi : (N, N) real array  – dielectric contrast  chi = eps_r - 1
    x, y : 1-D coordinate vectors
    """
    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y)
    chi = np.zeros((N, N), dtype=np.float64)

    # cylinder 1  – centre (+5 cm, +3 cm), radius 5 cm, chi=0.5
    mask1 = (X - 0.05) ** 2 + (Y - 0.03) ** 2 < 0.05 ** 2
    chi[mask1] = 0.5

    # cylinder 2  – centre (-6 cm, -2 cm), radius 3 cm, chi=1.0
    mask2 = (X + 0.06) ** 2 + (Y + 0.02) ** 2 < 0.03 ** 2
    chi[mask2] = 1.0

    return chi, x, y


# ====================================================================
# 2. Green's function  (2-D free-space, outgoing)
# ====================================================================
def green2d(k0, r):
    """G(r) = (j/4) H_0^{(2)}(k0 r)  – 2-D scalar Green's function."""
    r_safe = np.where(r < 1e-12, 1e-12, r)
    return (1j / 4.0) * hankel2(0, k0 * r_safe)


# ====================================================================
# 3. Sensing matrix  (Born linearisation)
# ====================================================================
def build_sensing_matrix(k0, gx, gy, tx_angles, rx_pos, ds):
    """
    A[l*N_RX + m, n] = k0^2  G(r_m, r_n)  E_inc(r_n, theta_l)  ds

    Parameters
    ----------
    gx, gy : 1-D grid coordinates
    tx_angles : (N_TX,) angles of plane-wave incidence
    rx_pos : (N_RX, 2) receiver positions
    ds : pixel area
    """
    Xg, Yg = np.meshgrid(gx, gy)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])  # (N^2, 2)
    n_pix = pts.shape[0]
    n_tx  = len(tx_angles)
    n_rx  = rx_pos.shape[0]

    A = np.zeros((n_tx * n_rx, n_pix), dtype=np.complex128)

    for l, theta in enumerate(tx_angles):
        # plane-wave direction
        d_hat = np.array([np.cos(theta), np.sin(theta)])
        # incident field on every pixel
        E_inc = np.exp(1j * k0 * (pts @ d_hat))            # (n_pix,)

        for m in range(n_rx):
            dr = np.sqrt((rx_pos[m, 0] - pts[:, 0]) ** 2
                       + (rx_pos[m, 1] - pts[:, 1]) ** 2)   # (n_pix,)
            G = green2d(k0, dr)
            A[l * n_rx + m, :] = k0 ** 2 * G * E_inc * ds

    return A


# ====================================================================
# 4. Forward operator
# ====================================================================
def forward_solve(A, chi_flat):
    """E_scat = A @ chi  (Born)."""
    return A @ chi_flat


# ====================================================================
# 5. Inverse solve  (Tikhonov)
# ====================================================================
def tikhonov_solve(A, y, lam="auto"):
    """Tikhonov inversion via SVD.

    If lam='auto', choose lambda via the Morozov discrepancy principle
    or a simple heuristic based on singular value spectrum.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    print(f"  SVD: {len(s)} singular values,  s_max={s[0]:.4e},  s_min={s[-1]:.4e}")

    if lam == "auto":
        # Truncated SVD approach: use singular values above noise floor
        # For 30dB SNR, noise_level ~ 10^{-1.5} ~ 0.03 relative
        noise_floor = s[0] * 10**(-SNR_DB/20)
        # Lambda = noise floor gives good balance
        lam = noise_floor
        print(f"  Noise floor = {noise_floor:.4e}")
        print(f"  Auto-lambda = {lam:.4e}")
        n_effective = np.sum(s > lam)
        print(f"  Effective rank = {n_effective}/{len(s)}")

    # Tikhonov filter factors
    filt = s / (s**2 + lam**2)
    # chi_hat = V diag(filt) U^H y
    chi_hat = Vh.conj().T @ (filt * (U.conj().T @ y))
    return chi_hat.real, lam


# ====================================================================
# 6. Metrics
# ====================================================================
def compute_psnr(gt, rec):
    peak = np.max(np.abs(gt))
    if peak == 0:
        return 0.0
    mse = np.mean((gt - rec) ** 2)
    if mse < 1e-30:
        return 100.0
    return 10.0 * np.log10(peak ** 2 / mse)


def compute_ssim(gt, rec):
    """Simplified SSIM for two 2-D images (float)."""
    from skimage.metrics import structural_similarity
    data_range = max(gt.max() - gt.min(), rec.max() - rec.min(), 1e-10)
    return structural_similarity(gt, rec, data_range=data_range)


def compute_rmse(gt, rec):
    return float(np.sqrt(np.mean((gt - rec) ** 2)))


# ====================================================================
# 7. Visualisation
# ====================================================================
def plot_results(chi_gt, chi_rec, gx, gy, metrics, save_paths):
    extent = [gx[0] * 1e3, gx[-1] * 1e3, gy[0] * 1e3, gy[-1] * 1e3]  # mm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(chi_gt, extent=extent, origin="lower",
                         cmap="jet", vmin=0, vmax=1.1)
    axes[0].set_title("Ground Truth  χ(r)")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(chi_rec, extent=extent, origin="lower",
                         cmap="jet", vmin=0, vmax=1.1)
    axes[1].set_title("Reconstructed  χ̂(r)")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)

    diff = np.abs(chi_gt - chi_rec)
    im2 = axes[2].imshow(diff, extent=extent, origin="lower", cmap="hot")
    axes[2].set_title("|Error|")
    axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("y [mm]")
    plt.colorbar(im2, ax=axes[2], shrink=0.85)

    fig.suptitle(
        f"EM Inverse Scattering (Born + Tikhonov)   "
        f"PSNR={metrics['PSNR']:.2f} dB   SSIM={metrics['SSIM']:.4f}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ====================================================================
# 8. Main pipeline
# ====================================================================
def main():
    t0 = time.time()
    sep = "=" * 70
    print(sep)
    print("EM Inverse Scattering — Born Approximation + Tikhonov")
    print(sep)

    # ── 1  phantom ──
    print("\n[1/6] Creating phantom …")
    chi_gt, gx, gy = create_phantom(N_GRID, DOMAIN_SIZE)
    ds = (gx[1] - gx[0]) * (gy[1] - gy[0])          # pixel area
    print(f"  Grid {N_GRID}×{N_GRID},  ds={ds:.4e} m²")

    # ── 2  array geometry ──
    print("\n[2/6] Array geometry …")
    tx_angles = np.linspace(0, 2 * np.pi, N_TX, endpoint=False)
    rx_angles = np.linspace(0, 2 * np.pi, N_RX, endpoint=False)
    rx_pos    = np.column_stack([R_ARRAY * np.cos(rx_angles),
                                 R_ARRAY * np.sin(rx_angles)])
    print(f"  {N_TX} TX,  {N_RX} RX  on circle R={R_ARRAY*100:.0f} cm")

    # ── 3  sensing matrix ──
    print("\n[3/6] Building sensing matrix (Born) …")
    A = build_sensing_matrix(K0, gx, gy, tx_angles, rx_pos, ds)
    print(f"  A shape = {A.shape}")

    # ── 4  forward + noise ──
    print("\n[4/6] Forward solve + noise …")
    chi_flat = chi_gt.ravel()
    y_clean  = forward_solve(A, chi_flat)
    noise_power = np.linalg.norm(y_clean) / (10 ** (SNR_DB / 20))
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(y_clean.shape)
             + 1j * rng.standard_normal(y_clean.shape)) / np.sqrt(2)
    noise *= noise_power / np.linalg.norm(noise)
    y_noisy = y_clean + noise
    print(f"  |y_clean| = {np.linalg.norm(y_clean):.4e}")
    print(f"  |noise|   = {np.linalg.norm(noise):.4e}")

    # ── 5  inverse solve ──
    print("\n[5/6] Tikhonov inversion …")
    chi_rec_flat, lam_used = tikhonov_solve(A, y_noisy, LAMBDA_REG)
    chi_rec = chi_rec_flat.reshape(N_GRID, N_GRID)
    print(f"  Lambda used = {lam_used:.4e}")
    # clip to physical range
    chi_rec = np.clip(chi_rec, 0.0, None)
    print(f"  χ̂ range = [{chi_rec.min():.4f}, {chi_rec.max():.4f}]")

    # ── 6  metrics ──
    print("\n[6/6] Metrics & visualisation …")
    psnr_val = compute_psnr(chi_gt, chi_rec)
    ssim_val = compute_ssim(chi_gt, chi_rec)
    rmse_val = compute_rmse(chi_gt, chi_rec)
    metrics = {"PSNR": psnr_val, "SSIM": ssim_val, "RMSE": rmse_val}
    print(f"  PSNR = {psnr_val:.2f} dB")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  RMSE = {rmse_val:.6f}")

    # save arrays
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), chi_gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), chi_rec)
    np.save(os.path.join(RESULTS_DIR, "scattered_field.npy"), y_noisy)
    np.save(os.path.join(RESULTS_DIR, "sensing_matrix.npy"), A)

    # website assets
    np.save(os.path.join(ASSETS_DIR, "gt_output.npy"), chi_gt)
    np.save(os.path.join(ASSETS_DIR, "recon_output.npy"), chi_rec)

    # metrics json
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # plots
    vis_paths = [
        os.path.join(RESULTS_DIR, "vis_result.png"),
        os.path.join(ASSETS_DIR, "vis_result.png"),
        os.path.join(WORKING_DIR, "vis_result.png"),
    ]
    plot_results(chi_gt, chi_rec, gx, gy, metrics, vis_paths)

    elapsed = time.time() - t0
    print(f"\n{sep}")
    print(f"DONE ({elapsed:.1f}s)  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
