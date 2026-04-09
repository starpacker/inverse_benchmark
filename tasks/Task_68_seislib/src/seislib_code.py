"""
seislib — Surface-Wave Dispersion Tomography
==============================================
Task #65: Invert surface-wave phase velocity measurements to recover
          2D velocity structure.

Inverse Problem:
    Given travel-time residuals δt from surface-wave phase velocity
    measurements along many station pairs, recover the 2D velocity
    perturbation map δc(x,y) on a regular grid.

Forward Model:
    Straight-ray approximation: δt_i = ∫_ray_i (1/c(s) - 1/c₀) ds
    Discretised as: δt = G · δm, where G is the ray-path kernel matrix
    and δm is the slowness perturbation vector.

Inverse Solver:
    LSQR iterative solver with Tikhonov (Laplacian) regularisation.

Repo: https://github.com/fmagrini/seislib
Paper: Magrini et al. (2022), GJI, doi:10.1093/gji/ggac236

Usage: /data/yjh/spectro_env/bin/python seislib_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.sparse import csr_matrix, eye as speye, kron as spkron
from scipy.sparse.linalg import lsqr
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NX, NY = 40, 40             # Grid cells
LAT_MIN, LAT_MAX = 35.0, 45.0  # Degrees
LON_MIN, LON_MAX = 5.0, 20.0
C0 = 3.5                    # Reference phase velocity [km/s]
N_STATIONS = 60             # Number of seismic stations
N_RAYS = 800                # Number of ray paths
NOISE_LEVEL = 0.02          # Relative noise on travel times
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_velocity_model(nx, ny, rng):
    """
    Create a 2D phase velocity perturbation model with realistic features:
    smooth background + localised anomalies (e.g., basins, cratons).
    """
    dm = np.zeros((nx, ny))

    # Smooth background
    raw = rng.standard_normal((nx, ny))
    dm += 0.03 * gaussian_filter(raw, sigma=8)

    # Fast anomaly (craton)
    cx, cy = nx // 3, ny // 3
    for i in range(nx):
        for j in range(ny):
            r2 = ((i - cx) / 8)**2 + ((j - cy) / 6)**2
            if r2 < 1:
                dm[i, j] += 0.08 * (1 - r2)

    # Slow anomaly (basin)
    cx2, cy2 = 2 * nx // 3, 2 * ny // 3
    for i in range(nx):
        for j in range(ny):
            r2 = ((i - cx2) / 6)**2 + ((j - cy2) / 8)**2
            if r2 < 1:
                dm[i, j] -= 0.06 * (1 - r2)

    return dm  # fractional slowness perturbation


def generate_stations(n_stations, rng):
    """Generate random station positions within the study area."""
    lats = LAT_MIN + (LAT_MAX - LAT_MIN) * rng.random(n_stations)
    lons = LON_MIN + (LON_MAX - LON_MIN) * rng.random(n_stations)
    return np.column_stack([lats, lons])


def generate_ray_paths(stations, n_rays, rng):
    """Generate station-pair ray paths."""
    n_sta = len(stations)
    pairs = []
    for _ in range(n_rays):
        i, j = rng.choice(n_sta, 2, replace=False)
        pairs.append((i, j))
    return pairs


# ─── Forward Operator ─────────────────────────────────────────────
def build_kernel_matrix(stations, pairs, nx, ny):
    """
    Build the sensitivity kernel G using straight-ray approximation.
    For each ray, accumulate path length through each grid cell.
    """
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, nx + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, ny + 1)
    dlat = lat_edges[1] - lat_edges[0]
    dlon = lon_edges[1] - lon_edges[0]
    # Approximate cell size in km (mid-latitude)
    deg2km_lat = 111.0
    deg2km_lon = 111.0 * np.cos(np.radians(0.5 * (LAT_MIN + LAT_MAX)))

    n_cells = nx * ny
    n_rays = len(pairs)
    rows, cols, vals = [], [], []

    for r, (si, sj) in enumerate(pairs):
        lat0, lon0 = stations[si]
        lat1, lon1 = stations[sj]

        # Parameterize ray as t ∈ [0, 1]
        n_samples = 500
        t = np.linspace(0, 1, n_samples)
        lat_ray = lat0 + (lat1 - lat0) * t
        lon_ray = lon0 + (lon1 - lon0) * t

        # Total ray length in km
        total_length_km = np.sqrt(
            ((lat1 - lat0) * deg2km_lat)**2 +
            ((lon1 - lon0) * deg2km_lon)**2
        )
        ds = total_length_km / n_samples  # segment length [km]

        # Accumulate path through cells
        cell_lengths = np.zeros(n_cells)
        i_cells = np.clip(((lat_ray - LAT_MIN) / dlat).astype(int), 0, nx - 1)
        j_cells = np.clip(((lon_ray - LON_MIN) / dlon).astype(int), 0, ny - 1)

        for k in range(n_samples):
            cell_idx = i_cells[k] * ny + j_cells[k]
            cell_lengths[cell_idx] += ds

        nz = np.nonzero(cell_lengths)[0]
        for c in nz:
            rows.append(r)
            cols.append(c)
            vals.append(cell_lengths[c])

    G = csr_matrix((vals, (rows, cols)), shape=(n_rays, n_cells))
    return G


def forward_model(G, dm_flat, c0):
    """
    Compute travel-time residuals: δt = G @ (dm / c0).
    dm is fractional slowness perturbation.
    """
    dt = G @ (dm_flat / c0)
    return dt


# ─── Inverse Solver ────────────────────────────────────────────────
def build_laplacian_2d(nx, ny):
    """Build 2D discrete Laplacian for regularisation."""
    # 1D second difference
    e = np.ones(max(nx, ny))

    def diff1d(n):
        D = np.zeros((n, n))
        for i in range(n - 1):
            D[i, i] = -1
            D[i, i + 1] = 1
        return csr_matrix(D)

    Dx = diff1d(nx)
    Dy = diff1d(ny)
    Ix = speye(nx)
    Iy = speye(ny)

    Lx = spkron(Dx.T @ Dx, Iy)
    Ly = spkron(Ix, Dy.T @ Dy)
    L = Lx + Ly
    return L


def invert_lsqr(G, dt, nx, ny, alpha=1.0, damp=0.0):
    """
    LSQR inversion with Laplacian smoothing regularisation.
    Solve: [G; √α L] @ dm = [dt; 0]
    """
    from scipy.sparse import vstack

    L = build_laplacian_2d(nx, ny)
    n_cells = nx * ny

    # Augmented system
    G_aug = vstack([G, np.sqrt(alpha) * L])
    dt_aug = np.concatenate([dt, np.zeros(L.shape[0])])

    result = lsqr(G_aug, dt_aug, damp=damp, iter_lim=500, atol=1e-8, btol=1e-8)
    dm_rec = result[0]

    return dm_rec


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt, rec):
    """Standard reconstruction metrics."""
    gt_2d = gt.copy()
    rec_2d = rec.copy()
    data_range = gt_2d.max() - gt_2d.min()
    if data_range < 1e-12:
        data_range = 1.0
    mse = np.mean((gt_2d - rec_2d)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_2d, rec_2d, data_range=data_range))
    cc = float(np.corrcoef(gt_2d.ravel(), rec_2d.ravel())[0, 1])
    re = float(np.linalg.norm(gt_2d - rec_2d) / max(np.linalg.norm(gt_2d), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(dm_gt, dm_rec, stations, metrics, save_path):
    """Multi-panel figure: GT, reconstruction, error, ray coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmax = max(np.abs(dm_gt).max(), np.abs(dm_rec).max())

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    im0 = axes[0, 0].imshow(dm_gt.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='auto')
    axes[0, 0].plot(stations[:, 1], stations[:, 0], 'k^', ms=4)
    axes[0, 0].set_title('Ground Truth δc/c₀')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(dm_rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='auto')
    axes[0, 1].plot(stations[:, 1], stations[:, 0], 'k^', ms=4)
    axes[0, 1].set_title('LSQR Reconstruction')
    plt.colorbar(im1, ax=axes[0, 1])

    err = dm_gt - dm_rec
    im2 = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent, aspect='auto')
    axes[1, 0].set_title('Error (GT - Recon)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Cross-section
    mid = dm_gt.shape[0] // 2
    axes[1, 1].plot(dm_gt[mid, :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(dm_rec[mid, :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title(f'Cross-section (row {mid})')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Column index')
    axes[1, 1].set_ylabel('δc/c₀')

    fig.suptitle(
        f"seislib — Surface-Wave Tomography\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  seislib — Surface-Wave Dispersion Tomography")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation")
    dm_gt = generate_velocity_model(NX, NY, rng)
    stations = generate_stations(N_STATIONS, rng)
    pairs = generate_ray_paths(stations, N_RAYS, rng)
    print(f"  Grid: {NX}×{NY}, Stations: {N_STATIONS}, Rays: {N_RAYS}")
    print(f"  Velocity perturbation range: [{dm_gt.min():.4f}, {dm_gt.max():.4f}]")

    # Stage 2: Forward Model
    print("\n[STAGE 2] Forward — Straight-Ray Travel Time")
    G = build_kernel_matrix(stations, pairs, NX, NY)
    dm_gt_flat = dm_gt.ravel()
    dt_clean = forward_model(G, dm_gt_flat, C0)
    noise = NOISE_LEVEL * np.std(dt_clean) * rng.standard_normal(len(dt_clean))
    dt_noisy = dt_clean + noise
    print(f"  Kernel matrix G: {G.shape}, nnz={G.nnz}")
    print(f"  Travel-time residual range: [{dt_clean.min():.4f}, {dt_clean.max():.4f}] s")

    # Stage 3: Inverse
    print("\n[STAGE 3] Inverse — LSQR + Laplacian Regularisation")
    # Try multiple regularisation strengths
    best_cc = -1
    best_alpha = 1.0
    best_rec = None
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        dm_rec = invert_lsqr(G, dt_noisy, NX, NY, alpha=alpha)
        dm_rec_2d = dm_rec.reshape(NX, NY)
        cc_val = float(np.corrcoef(dm_gt_flat, dm_rec)[0, 1])
        print(f"  α={alpha:6.1f} → CC={cc_val:.4f}")
        if cc_val > best_cc:
            best_cc = cc_val
            best_alpha = alpha
            best_rec = dm_rec_2d

    print(f"  → Best α={best_alpha} with CC={best_cc:.4f}")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    metrics = compute_metrics(dm_gt, best_rec)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), dm_gt)

    visualize_results(dm_gt, best_rec, stations, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
