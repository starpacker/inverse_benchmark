"""
straintool_geo - Geodetic Strain Rate Inversion
=================================================
From GNSS velocity observations at discrete stations, invert for the continuous
crustal strain rate tensor field using weighted least-squares collocation.

Physics:
  - Forward: v(x_i) = ε(x_0) × (x_i - x_0) + ω(x_0) × (x_i - x_0) + t(x_0)
  - ε: strain rate tensor (2×2 symmetric), ω: rotation rate, t: translation
  - Inverse: Local weighted least squares with Gaussian spatial weighting
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from skimage.metrics import structural_similarity as ssim

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_106_straintool_geo"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
N_STATIONS     = 500         # number of GNSS stations
REGION_SIZE    = 500.0       # km — region side length
GRID_N         = 20          # grid resolution for strain output
NOISE_LEVEL    = 0.2         # mm/yr velocity noise std
GAUSS_SIGMA    = 30.0        # km — Gaussian weighting width
MIN_STATIONS   = 10          # minimum stations for local inversion
SEED           = 42

np.random.seed(SEED)


# ====================================================================
# 1. Ground truth strain rate field
# ====================================================================
def create_gt_strain_field(grid_x, grid_y):
    """
    Create synthetic strain rate tensor field with a localized
    deformation zone (e.g., shear zone / plate boundary).

    Returns exx, exy, eyy fields in units of nanostrain/yr.
    """
    X, Y = np.meshgrid(grid_x, grid_y)

    # Background strain (regional extension)
    exx_bg = 10.0  # nanostrain/yr
    eyy_bg = -5.0
    exy_bg = 2.0

    exx = np.full_like(X, exx_bg)
    eyy = np.full_like(X, eyy_bg)
    exy = np.full_like(X, exy_bg)

    # Localized shear zone (fault-like feature)
    fault_x = REGION_SIZE * 0.5
    fault_y = REGION_SIZE * 0.5
    fault_width = 60.0  # km

    # Gaussian shear anomaly centered on fault
    anomaly = np.exp(-((X - fault_x)**2 + (Y - fault_y)**2) / (2 * fault_width**2))

    exx += 80.0 * anomaly   # compression across zone
    eyy += -40.0 * anomaly  # extension along zone
    exy += 120.0 * anomaly  # shear across zone

    # Secondary deformation zone
    anomaly2 = np.exp(-((X - REGION_SIZE * 0.2)**2 + (Y - REGION_SIZE * 0.7)**2)
                      / (2 * 40.0**2))
    exx += -30.0 * anomaly2
    eyy += 50.0 * anomaly2
    exy += -60.0 * anomaly2

    return exx, exy, eyy


# ====================================================================
# 2. Forward model: velocity from strain
# ====================================================================
def forward_velocity(station_x, station_y, ref_x, ref_y, exx, exy, eyy,
                     omega=0.0, tx=0.0, ty=0.0):
    """
    Compute velocity at a station from local strain rate tensor.
    v(x) = ε × (x - x_0) + ω × (x - x_0) + t

    v_x = exx*(x-x0) + exy*(y-y0) - omega*(y-y0) + tx
    v_y = exy*(x-x0) + eyy*(y-y0) + omega*(x-x0) + ty
    """
    dx = station_x - ref_x
    dy = station_y - ref_y

    vx = exx * dx + exy * dy - omega * dy + tx
    vy = exy * dx + eyy * dy + omega * dx + ty

    return vx, vy


def generate_synthetic_velocities(stations, grid_x, grid_y, exx, exy, eyy):
    """
    Generate synthetic GNSS velocities from the displacement field
    computed by integrating the strain field on a fine grid.

    Uses separate x- and y-integrations with cross-term corrections:
      vx(x,y) = ∫_{x0}^x εxx(x',y) dx' + ∫_{y0}^y εxy(x0,y') dy'
      vy(x,y) = ∫_{y0}^y εyy(x,y') dy' + ∫_{x0}^x εxy(x',y0) dx'
    
    This gives exact ∂vx/∂x = εxx and ∂vy/∂y = εyy.
    Cross-strain εxy recovery depends on its spatial smoothness.
    """
    from scipy.interpolate import RegularGridInterpolator

    n_fine = 100
    fine_x = np.linspace(grid_x.min(), grid_x.max(), n_fine)
    fine_y = np.linspace(grid_y.min(), grid_y.max(), n_fine)

    interp_exx = RegularGridInterpolator((grid_y, grid_x), exx,
                                         bounds_error=False, fill_value=None)
    interp_exy = RegularGridInterpolator((grid_y, grid_x), exy,
                                         bounds_error=False, fill_value=None)
    interp_eyy = RegularGridInterpolator((grid_y, grid_x), eyy,
                                         bounds_error=False, fill_value=None)

    FX, FY = np.meshgrid(fine_x, fine_y)
    pts_fine = np.column_stack([FY.ravel(), FX.ravel()])

    exx_fine = interp_exx(pts_fine).reshape(n_fine, n_fine)
    exy_fine = interp_exy(pts_fine).reshape(n_fine, n_fine)
    eyy_fine = interp_eyy(pts_fine).reshape(n_fine, n_fine)

    scale = 1e-3
    cx = n_fine // 2
    cy = n_fine // 2
    ddx = fine_x[1] - fine_x[0]
    ddy = fine_y[1] - fine_y[0]

    vx_grid = np.zeros((n_fine, n_fine))
    vy_grid = np.zeros((n_fine, n_fine))

    # --- vx(x,y) = ∫_{x0}^x εxx(x',y) dx' + ∫_{y0}^y εxy(x0,y') dy' ---
    # Part 1: εxx along x for each y row
    for j in range(n_fine):
        cum = 0.0
        for i in range(cx + 1, n_fine):
            cum += exx_fine[j, i] * ddx * scale
            vx_grid[j, i] = cum
        cum = 0.0
        for i in range(cx - 1, -1, -1):
            cum -= exx_fine[j, i + 1] * ddx * scale
            vx_grid[j, i] = cum

    # Part 2: εxy along y at x=x0 (adds y-offset)
    vx_cross = np.zeros(n_fine)
    cum = 0.0
    for j in range(cy + 1, n_fine):
        cum += exy_fine[j, cx] * ddy * scale
        vx_cross[j] = cum
    cum = 0.0
    for j in range(cy - 1, -1, -1):
        cum -= exy_fine[j + 1, cx] * ddy * scale
        vx_cross[j] = cum

    for j in range(n_fine):
        vx_grid[j, :] += vx_cross[j]

    # --- vy(x,y) = ∫_{y0}^y εyy(x,y') dy' + ∫_{x0}^x εxy(x',y0) dx' ---
    # Part 1: εyy along y for each x column
    for i in range(n_fine):
        cum = 0.0
        for j in range(cy + 1, n_fine):
            cum += eyy_fine[j, i] * ddy * scale
            vy_grid[j, i] = cum
        cum = 0.0
        for j in range(cy - 1, -1, -1):
            cum -= eyy_fine[j + 1, i] * ddy * scale
            vy_grid[j, i] = cum

    # Part 2: εxy along x at y=y0 (adds x-offset)
    vy_cross = np.zeros(n_fine)
    cum = 0.0
    for i in range(cx + 1, n_fine):
        cum += exy_fine[cy, i] * ddx * scale
        vy_cross[i] = cum
    cum = 0.0
    for i in range(cx - 1, -1, -1):
        cum -= exy_fine[cy, i + 1] * ddx * scale
        vy_cross[i] = cum

    for i in range(n_fine):
        vy_grid[:, i] += vy_cross[i]

    # Interpolate velocities to station locations
    interp_vx = RegularGridInterpolator((fine_y, fine_x), vx_grid,
                                         bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((fine_y, fine_x), vy_grid,
                                         bounds_error=False, fill_value=0.0)

    n = stations.shape[0]
    vx_out = np.array([float(interp_vx((s[1], s[0]))) for s in stations])
    vy_out = np.array([float(interp_vy((s[1], s[0]))) for s in stations])

    vx_out += NOISE_LEVEL * np.random.randn(n)
    vy_out += NOISE_LEVEL * np.random.randn(n)

    return vx_out, vy_out


# ====================================================================
# 3. Inverse: Weighted least-squares strain estimation
# ====================================================================
def invert_strain_at_point(px, py, stations, vx, vy, sigma):
    """
    Estimate strain rate tensor at point (px, py) using Gaussian-weighted
    velocity gradient estimation.

    Fits a linear model for each velocity component:
      vx(x,y) = a0 + a1*(x-px) + a2*(y-py)
      vy(x,y) = b0 + b1*(x-px) + b2*(y-py)

    Then: exx = a1/scale, exy = (a2 + b1)/(2*scale), eyy = b2/scale
    """
    n = stations.shape[0]

    # Gaussian weights centered on the query point
    dist = np.sqrt((stations[:, 0] - px)**2 + (stations[:, 1] - py)**2)
    weights = np.exp(-dist**2 / (2 * sigma**2))

    # Only use stations with significant weight
    mask = weights > 0.01 * weights.max()
    if np.sum(mask) < MIN_STATIONS:
        idx = np.argsort(dist)[:MIN_STATIONS]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True

    w = weights[mask]
    sx = stations[mask, 0]
    sy = stations[mask, 1]
    vx_sel = vx[mask]
    vy_sel = vy[mask]

    n_sel = len(w)

    # Displacement from query point
    dx = sx - px
    dy = sy - py

    # Build design matrix for linear fit: v = a0 + a1*dx + a2*dy
    A = np.column_stack([np.ones(n_sel), dx, dy])

    # Weighted least squares for vx
    W = np.diag(w)
    AW = A.T @ W
    try:
        reg = 1e-10 * np.eye(3)
        ax = np.linalg.solve(AW @ A + reg, AW @ vx_sel)
        ay = np.linalg.solve(AW @ A + reg, AW @ vy_sel)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0

    # Scale factor
    scale = 1e-3

    # Extract strain from velocity gradients
    # dvx/dx = a1 = exx * scale → exx = a1 / scale
    # dvx/dy = a2 = exy * scale → exy_from_vx = a2 / scale
    # dvy/dx = b1 = exy * scale → exy_from_vy = b1 / scale
    # dvy/dy = b2 = eyy * scale → eyy = b2 / scale
    exx = ax[1] / scale
    eyy = ay[2] / scale
    exy = 0.5 * (ax[2] + ay[1]) / scale  # symmetric strain

    return exx, exy, eyy


def invert_strain_field(grid_x, grid_y, stations, vx, vy, sigma):
    """Invert for strain rate field on a grid."""
    ny = len(grid_y)
    nx = len(grid_x)
    exx_rec = np.zeros((ny, nx))
    exy_rec = np.zeros((ny, nx))
    eyy_rec = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            exx_r, exy_r, eyy_r = invert_strain_at_point(
                grid_x[i], grid_y[j], stations, vx, vy, sigma
            )
            exx_rec[j, i] = exx_r
            exy_rec[j, i] = exy_r
            eyy_rec[j, i] = eyy_r

    return exx_rec, exy_rec, eyy_rec


# ====================================================================
# 4. Metrics
# ====================================================================
def compute_field_metrics(gt, rec):
    """Compute PSNR, SSIM, CC for 2D field comparison."""
    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-15)
    rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-15)

    # PSNR
    mse = np.mean((gt_n - rec_n)**2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0

    # SSIM
    data_range = max(gt_n.max() - gt_n.min(), rec_n.max() - rec_n.min())
    if data_range < 1e-15:
        data_range = 1.0
    ssim_val = ssim(gt_n, rec_n, data_range=data_range)

    # CC
    gt_z = gt_n - gt_n.mean()
    rec_z = rec_n - rec_n.mean()
    denom = np.sqrt(np.sum(gt_z**2) * np.sum(rec_z**2))
    cc = np.sum(gt_z * rec_z) / denom if denom > 1e-15 else 0.0

    return float(psnr), float(ssim_val), float(cc)


# ====================================================================
# 5. Visualization
# ====================================================================
def plot_results(grid_x, grid_y, gt_exx, gt_exy, gt_eyy,
                 rec_exx, rec_exy, rec_eyy,
                 stations, vx, vy, metrics):
    """Visualize strain rate maps and velocity arrows."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]

    components = [
        ("εxx", gt_exx, rec_exx),
        ("εxy", gt_exy, rec_exy),
        ("εyy", gt_eyy, rec_eyy),
    ]

    for row, (name, gt, rec) in enumerate(components):
        vmin = min(gt.min(), rec.min())
        vmax = max(gt.max(), rec.max())

        ax = axes[row, 0]
        im = ax.imshow(gt, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.scatter(stations[:, 0], stations[:, 1], c='k', s=10, marker='^')
        ax.set_title(f"GT {name} (nanostrain/yr)", fontsize=12)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax)

        ax = axes[row, 1]
        im = ax.imshow(rec, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.scatter(stations[:, 0], stations[:, 1], c='k', s=10, marker='^')
        m_comp = metrics[f'{name}']
        ax.set_title(f"Reconstructed {name}\nPSNR={m_comp['PSNR']:.1f}dB, "
                     f"SSIM={m_comp['SSIM']:.3f}, CC={m_comp['CC']:.3f}", fontsize=11)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ====================================================================
# 6. Main
# ====================================================================
def main():
    print("=" * 60)
    print("Task 106: Geodetic Strain Rate Inversion")
    print("=" * 60)

    # 1) Grid
    grid_x = np.linspace(20, REGION_SIZE - 20, GRID_N)
    grid_y = np.linspace(20, REGION_SIZE - 20, GRID_N)
    print(f"[1] Grid: {GRID_N}×{GRID_N} over {REGION_SIZE}×{REGION_SIZE} km")

    # 2) Ground truth strain field
    print("[2] Creating ground truth strain field ...")
    gt_exx, gt_exy, gt_eyy = create_gt_strain_field(grid_x, grid_y)
    print(f"    εxx range: [{gt_exx.min():.1f}, {gt_exx.max():.1f}] nanostrain/yr")
    print(f"    εxy range: [{gt_exy.min():.1f}, {gt_exy.max():.1f}] nanostrain/yr")
    print(f"    εyy range: [{gt_eyy.min():.1f}, {gt_eyy.max():.1f}] nanostrain/yr")

    # 3) Station positions
    print("[3] Generating GNSS stations ...")
    stations = np.column_stack([
        np.random.uniform(30, REGION_SIZE - 30, N_STATIONS),
        np.random.uniform(30, REGION_SIZE - 30, N_STATIONS),
    ])
    print(f"    {N_STATIONS} stations")

    # 4) Synthetic velocities
    print("[4] Computing synthetic velocities ...")
    vx, vy = generate_synthetic_velocities(stations, grid_x, grid_y, gt_exx, gt_exy, gt_eyy)
    print(f"    Vx range: [{vx.min():.2f}, {vx.max():.2f}] mm/yr")
    print(f"    Vy range: [{vy.min():.2f}, {vy.max():.2f}] mm/yr")

    # 5) Inverse
    print(f"[5] Inverting strain field (σ={GAUSS_SIGMA} km) ...")
    t0 = time.time()
    rec_exx, rec_exy, rec_eyy = invert_strain_field(
        grid_x, grid_y, stations, vx, vy, GAUSS_SIGMA
    )
    t_inv = time.time() - t0
    print(f"    Inversion: {t_inv:.1f}s")

    # 6) Metrics per component
    print("[6] Computing metrics ...")
    comp_metrics = {}
    all_psnr, all_ssim, all_cc = [], [], []

    for name, gt, rec in [("εxx", gt_exx, rec_exx),
                          ("εxy", gt_exy, rec_exy),
                          ("εyy", gt_eyy, rec_eyy)]:
        p, s, c = compute_field_metrics(gt, rec)
        comp_metrics[name] = {"PSNR": p, "SSIM": s, "CC": c}
        all_psnr.append(p)
        all_ssim.append(s)
        all_cc.append(c)
        print(f"    {name}: PSNR={p:.2f}, SSIM={s:.4f}, CC={c:.4f}")

    avg_psnr = float(np.mean(all_psnr))
    avg_ssim = float(np.mean(all_ssim))
    avg_cc = float(np.mean(all_cc))
    print(f"\n    Average: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, CC={avg_cc:.4f}")

    metrics = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "CC": avg_cc,
        "components": comp_metrics,
    }

    # 7) Save
    print("[7] Saving outputs ...")
    gt_all = np.stack([gt_exx, gt_exy, gt_eyy])
    rec_all = np.stack([rec_exx, rec_exy, rec_eyy])

    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_all)
        np.save(os.path.join(d, "recon_output.npy"), rec_all)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # 8) Plot
    print("[8] Plotting ...")
    plot_results(grid_x, grid_y, gt_exx, gt_exy, gt_eyy,
                 rec_exx, rec_exy, rec_eyy,
                 stations, vx, vy, comp_metrics)

    print(f"\n{'=' * 60}")
    print("Task 106 COMPLETE")
    print(f"{'=' * 60}")
    return metrics


if __name__ == "__main__":
    metrics = main()
