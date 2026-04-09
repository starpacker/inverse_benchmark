"""
zdipy_doppler — Doppler Imaging of Stellar Surfaces
=====================================================
Reconstruct stellar surface brightness from rotationally-broadened spectral
line profiles observed at multiple rotation phases.

Physics:
  - A rapidly rotating star has surface elements Doppler-shifted by
    v_j(phi) = v_eq * sin(i) * cos(lat_j) * sin(lon_j + 2*pi*phi)
  - Each visible element contributes a local line profile weighted by
    its brightness and projected area (limb-darkening via mu = cos theta)
  - Forward: I(v, phi) = Sum_j  B_j * g(v - v_j(phi)) * mu_j * dOmega_j * vis_j
  - Inverse: Tikhonov-regularised least-squares to recover brightness map B_j
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_98_zdipy_doppler"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── stellar / observation parameters ──────────────────────────────
N_LAT       = 20          # latitude zones
N_LON       = 40          # longitude bins
N_PIX       = N_LAT * N_LON   # 800 surface elements
V_EQ        = 50.0        # equatorial rotation velocity (km/s)
INCLINATION = 60.0        # stellar inclination (degrees)
N_PHASES    = 32          # number of rotation phases observed
N_VBINS     = 100         # velocity bins in the line profile
V_MAX       = 70.0        # velocity axis range +/- V_MAX (km/s)
LOCAL_WIDTH = 5.0         # local line profile Gaussian width (km/s)
LIMB_DARK   = 0.6         # linear limb-darkening coefficient epsilon
SNR_DB      = 30.0        # signal-to-noise ratio (dB)
LAMBDA_REG  = "auto"      # Tikhonov regularisation parameter


# ====================================================================
# 1. Surface grid (latitude-longitude)
# ====================================================================
def create_surface_grid(n_lat, n_lon):
    """Equal-area-ish lat/lon grid on the stellar surface.

    Returns
    -------
    lats    : (N_PIX,)  latitudes in radians [-pi/2, pi/2]
    lons    : (N_PIX,)  longitudes in radians [0, 2*pi)
    d_omega : (N_PIX,)  solid-angle element for each pixel
    """
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lon_edges = np.linspace(0, 2 * np.pi, n_lon + 1)

    lats_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lons_1d = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    LONS, LATS = np.meshgrid(lons_1d, lats_1d)  # (n_lat, n_lon)
    lats = LATS.ravel()
    lons = LONS.ravel()

    # solid-angle element  dOmega = cos(lat) * d_lat * d_lon
    d_lat = lat_edges[1] - lat_edges[0]
    d_lon = lon_edges[1] - lon_edges[0]
    d_omega = np.abs(np.cos(lats)) * d_lat * d_lon

    return lats, lons, d_omega


# ====================================================================
# 2. Ground-truth brightness map (star with dark spots)
# ====================================================================
def create_ground_truth(lats, lons):
    """Create a brightness map with 3 dark spots on a bright photosphere.

    Returns
    -------
    B_gt : (N_PIX,) surface brightness (1 = photosphere, <1 = spot)
    """
    B_gt = np.ones(len(lats), dtype=np.float64)

    # Spot 1: large cool spot near equator
    lat_c1, lon_c1 = np.radians(10), np.radians(90)
    r1, depth1 = np.radians(20), 0.3
    ang1 = np.arccos(np.clip(
        np.sin(lats) * np.sin(lat_c1) +
        np.cos(lats) * np.cos(lat_c1) * np.cos(lons - lon_c1), -1, 1))
    B_gt[ang1 < r1] = depth1

    # Spot 2: mid-latitude spot
    lat_c2, lon_c2 = np.radians(40), np.radians(220)
    r2, depth2 = np.radians(15), 0.4
    ang2 = np.arccos(np.clip(
        np.sin(lats) * np.sin(lat_c2) +
        np.cos(lats) * np.cos(lat_c2) * np.cos(lons - lon_c2), -1, 1))
    B_gt[ang2 < r2] = depth2

    # Spot 3: small polar spot
    lat_c3, lon_c3 = np.radians(65), np.radians(350)
    r3, depth3 = np.radians(12), 0.5
    ang3 = np.arccos(np.clip(
        np.sin(lats) * np.sin(lat_c3) +
        np.cos(lats) * np.cos(lat_c3) * np.cos(lons - lon_c3), -1, 1))
    B_gt[ang3 < r3] = depth3

    return B_gt


# ====================================================================
# 3. Visibility & projected velocity
# ====================================================================
def compute_visibility_and_velocity(lats, lons, phase, inc_rad, v_eq):
    """For a given rotation phase, compute visibility and radial velocity.

    Parameters
    ----------
    phase : float in [0, 1)  rotation phase

    Returns
    -------
    mu    : (N_PIX,)  cos(angle to observer);  >0 means visible
    v_rad : (N_PIX,)  radial velocity of each surface element (km/s)
    """
    # At phase phi the star rotates, equivalent to shifting longitudes
    shifted_lon = lons + 2.0 * np.pi * phase

    # Direction cosine: mu = sin(i)*cos(lat)*cos(shifted_lon) + cos(i)*sin(lat)
    sin_i = np.sin(inc_rad)
    cos_i = np.cos(inc_rad)
    mu = sin_i * np.cos(lats) * np.cos(shifted_lon) + cos_i * np.sin(lats)

    # Radial velocity: projection of rotation onto line of sight
    # v_rad = v_eq * sin(i) * cos(lat) * sin(shifted_lon)
    v_rad = v_eq * sin_i * np.cos(lats) * np.sin(shifted_lon)

    return mu, v_rad


# ====================================================================
# 4. Build design matrix  (Forward model)
# ====================================================================
def build_design_matrix(lats, lons, d_omega, phases, v_axis, v_eq, inc_deg,
                        local_width, limb_eps):
    """Assemble the design matrix A so that  d = A @ B.

    d : (N_PHASES * N_VBINS,)  observed line profile data vector
    B : (N_PIX,)               surface brightness

    A[k, j] = vis_j(phi) * mu_j(phi)^ld * dOmega_j * g(v_k - v_j(phi))
    where g is a Gaussian local line profile, k indexes (phase, v-bin).
    """
    inc_rad = np.radians(inc_deg)
    n_ph = len(phases)
    n_v  = len(v_axis)
    n_pix = len(lats)

    A = np.zeros((n_ph * n_v, n_pix), dtype=np.float64)

    for ip, phi in enumerate(phases):
        mu, v_rad = compute_visibility_and_velocity(lats, lons, phi, inc_rad, v_eq)
        visible = mu > 0.0

        # Limb-darkening weight:  I_ld = 1 - eps*(1 - mu)
        ld_weight = np.where(visible, 1.0 - limb_eps * (1.0 - mu), 0.0)
        weight = ld_weight * d_omega * visible.astype(np.float64)

        for j in range(n_pix):
            if not visible[j]:
                continue
            # Gaussian local line profile centred at v_rad[j]
            profile = np.exp(-0.5 * ((v_axis - v_rad[j]) / local_width) ** 2)
            profile /= (local_width * np.sqrt(2.0 * np.pi))
            A[ip * n_v:(ip + 1) * n_v, j] = weight[j] * profile

    return A


# ====================================================================
# 5. Forward solve
# ====================================================================
def forward_solve(A, B):
    """Compute observed line profiles  d = A @ B."""
    return A @ B


# ====================================================================
# 6. Tikhonov inverse solve
# ====================================================================
def tikhonov_solve(A, d, lam):
    """Solve  min ||A B - d||^2 + lambda ||B - 1||^2   (default brightness = 1).

    Using the substitution  B' = B - 1:
        min ||A B' - (d - A*1)||^2 + lambda*||B'||^2

    Returns
    -------
    B_rec    : (N_PIX,)  reconstructed brightness
    lam_used : float
    """
    n_pix = A.shape[1]
    ones_vec = np.ones(n_pix)
    d_shifted = d - A @ ones_vec  # shifted data

    AtA = A.T @ A
    Atd = A.T @ d_shifted

    if lam == "auto":
        # Heuristic: lambda = trace(AtA) / n_pix * factor
        lam_used = np.trace(AtA) / n_pix * 0.3
    else:
        lam_used = float(lam)

    B_prime = np.linalg.solve(AtA + lam_used * np.eye(n_pix), Atd)
    B_rec = B_prime + ones_vec

    return B_rec, lam_used


# ====================================================================
# 7. Metrics
# ====================================================================
def compute_psnr(gt, rec):
    """Peak signal-to-noise ratio."""
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    mse = np.mean((gt - rec) ** 2)
    if mse < 1e-30:
        return 100.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim(gt, rec, C1=None, C2=None):
    """Simple SSIM between 1-D or 2-D arrays."""
    gt_f = gt.ravel().astype(np.float64)
    rec_f = rec.ravel().astype(np.float64)
    drange = gt_f.max() - gt_f.min()
    if C1 is None:
        C1 = (0.01 * drange) ** 2
    if C2 is None:
        C2 = (0.03 * drange) ** 2
    mu_x = gt_f.mean()
    mu_y = rec_f.mean()
    sig_x = gt_f.std()
    sig_y = rec_f.std()
    sig_xy = np.mean((gt_f - mu_x) * (rec_f - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2)
    return float(num / den)


def compute_cc(gt, rec):
    """Pearson correlation coefficient."""
    gt_f = gt.ravel().astype(np.float64)
    rec_f = rec.ravel().astype(np.float64)
    gt_z = gt_f - gt_f.mean()
    rec_z = rec_f - rec_f.mean()
    denom = np.linalg.norm(gt_z) * np.linalg.norm(rec_z)
    if denom < 1e-30:
        return 0.0
    return float(np.dot(gt_z, rec_z) / denom)


def compute_rmse(gt, rec):
    """Root mean square error."""
    return float(np.sqrt(np.mean((gt - rec) ** 2)))


# ====================================================================
# 8. Visualisation
# ====================================================================
def plot_results(B_gt, B_rec, n_lat, n_lon, metrics, save_paths):
    """Plot ground truth vs reconstruction as Mollweide maps."""
    B_gt_2d  = B_gt.reshape(n_lat, n_lon)
    B_rec_2d = B_rec.reshape(n_lat, n_lon)

    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lon_edges = np.linspace(0, 2 * np.pi, n_lon + 1)
    lats_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lons_1d = 0.5 * (lon_edges[:-1] + lon_edges[1:]) - np.pi  # shift to [-pi, pi]

    LON, LAT = np.meshgrid(lons_1d, lats_1d)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             subplot_kw={"projection": "mollweide"})

    # GT
    im0 = axes[0].pcolormesh(LON, LAT, B_gt_2d, cmap="inferno",
                             vmin=0, vmax=1, shading="auto")
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.05, shrink=0.8)

    # Reconstruction
    im1 = axes[1].pcolormesh(LON, LAT, np.clip(B_rec_2d, 0, 1), cmap="inferno",
                             vmin=0, vmax=1, shading="auto")
    axes[1].set_title("Reconstruction", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.05, shrink=0.8)

    # Residual
    residual = np.abs(B_gt_2d - B_rec_2d)
    im2 = axes[2].pcolormesh(LON, LAT, residual, cmap="hot",
                             vmin=0, vmax=0.5, shading="auto")
    axes[2].set_title("|Residual|", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.05, shrink=0.8)

    fig.suptitle(
        f"Doppler Imaging — Stellar Surface Brightness Recovery\n"
        f"PSNR={metrics['PSNR']:.2f} dB   SSIM={metrics['SSIM']:.4f}   "
        f"CC={metrics['CC']:.4f}",
        fontsize=13, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ====================================================================
# 9. Main pipeline
# ====================================================================
def main():
    t0 = time.time()
    sep = "=" * 70
    print(sep)
    print("Doppler Imaging — Stellar Surface Brightness Recovery")
    print(sep)

    # ── 1  surface grid ──
    print("\n[1/7] Creating surface grid ...")
    lats, lons, d_omega = create_surface_grid(N_LAT, N_LON)
    print(f"  {N_LAT} lat x {N_LON} lon = {N_PIX} surface elements")

    # ── 2  ground truth ──
    print("\n[2/7] Ground-truth brightness map (3 dark spots) ...")
    B_gt = create_ground_truth(lats, lons)
    n_spot = int(np.sum(B_gt < 0.9))
    print(f"  Photosphere=1.0,  {n_spot} spot pixels")
    print(f"  B range = [{B_gt.min():.2f}, {B_gt.max():.2f}]")

    # ── 3  observation setup ──
    print("\n[3/7] Observation setup ...")
    phases = np.linspace(0, 1.0, N_PHASES, endpoint=False)
    v_axis = np.linspace(-V_MAX, V_MAX, N_VBINS)
    print(f"  {N_PHASES} phases,  {N_VBINS} velocity bins  +/-{V_MAX} km/s")
    print(f"  v_eq={V_EQ} km/s,  inc={INCLINATION} deg,  eps_LD={LIMB_DARK}")

    # ── 4  design matrix ──
    print("\n[4/7] Building design matrix ...")
    A = build_design_matrix(lats, lons, d_omega, phases, v_axis,
                            V_EQ, INCLINATION, LOCAL_WIDTH, LIMB_DARK)
    print(f"  A shape = {A.shape}  (data={N_PHASES * N_VBINS}, pixels={N_PIX})")
    print(f"  A nonzero fraction = {np.count_nonzero(A) / A.size:.4f}")

    # ── 5  forward + noise ──
    print("\n[5/7] Forward solve + noise ...")
    d_clean = forward_solve(A, B_gt)
    noise_power = np.linalg.norm(d_clean) / (10 ** (SNR_DB / 20))
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(d_clean.shape)
    noise *= noise_power / np.linalg.norm(noise)
    d_noisy = d_clean + noise
    print(f"  |d_clean| = {np.linalg.norm(d_clean):.6e}")
    print(f"  |noise|   = {np.linalg.norm(noise):.6e}")
    print(f"  SNR       = {SNR_DB} dB")

    # ── 6  inverse solve ──
    print("\n[6/7] Tikhonov inversion ...")
    B_rec, lam_used = tikhonov_solve(A, d_noisy, LAMBDA_REG)
    # Clip to physical range [0, 1]
    B_rec = np.clip(B_rec, 0.0, 1.0)
    print(f"  Lambda = {lam_used:.6e}")
    print(f"  B_rec range = [{B_rec.min():.4f}, {B_rec.max():.4f}]")

    # ── 7  metrics ──
    print("\n[7/7] Metrics & visualisation ...")
    psnr_val = compute_psnr(B_gt, B_rec)
    ssim_val = compute_ssim(B_gt, B_rec)
    cc_val   = compute_cc(B_gt, B_rec)
    rmse_val = compute_rmse(B_gt, B_rec)
    metrics = {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "CC":   cc_val,
        "RMSE": rmse_val,
    }
    print(f"  PSNR = {psnr_val:.2f} dB")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  CC   = {cc_val:.4f}")
    print(f"  RMSE = {rmse_val:.6f}")

    # ── save arrays ──
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), B_gt.reshape(N_LAT, N_LON))
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), B_rec.reshape(N_LAT, N_LON))
    np.save(os.path.join(RESULTS_DIR, "data_noisy.npy"), d_noisy.reshape(N_PHASES, N_VBINS))
    np.save(os.path.join(RESULTS_DIR, "design_matrix.npy"), A)

    # website assets
    np.save(os.path.join(ASSETS_DIR, "gt_output.npy"), B_gt.reshape(N_LAT, N_LON))
    np.save(os.path.join(ASSETS_DIR, "recon_output.npy"), B_rec.reshape(N_LAT, N_LON))

    # metrics JSON
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
    plot_results(B_gt, B_rec, N_LAT, N_LON, metrics, vis_paths)

    elapsed = time.time() - t0
    print(f"\n{sep}")
    print(f"DONE ({elapsed:.1f}s)  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}  CC={cc_val:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
