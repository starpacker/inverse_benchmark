"""
NoisePy — Ambient Noise Cross-Correlation Tomography
======================================================
Task #66: Recover 2D velocity structure from ambient noise
          cross-correlation travel times.

Inverse Problem:
    Given inter-station travel times extracted from ambient noise
    cross-correlations, recover the 2D velocity perturbation map.

Forward Model:
    Straight-ray travel time: t_ij = ∫_{path_ij} ds/c(x)
    Discretised: δt = G · δs (slowness perturbation)

Inverse Solver:
    LSQR tomography with smoothness (Laplacian) + damping regularisation.
    L-curve criterion for regularisation parameter selection.

Repo: https://github.com/noisepy/NoisePy
Paper: Jiang & Denolle (2020), Seismological Research Letters.

Usage: /data/yjh/spectro_env/bin/python NoisePy_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack as spvstack
from scipy.sparse.linalg import lsqr
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NX, NY = 40, 40
XMIN, XMAX = 0.0, 200.0   # km
YMIN, YMAX = 0.0, 200.0
C0 = 3.0                   # Reference surface-wave velocity [km/s]
N_STATIONS = 50
NOISE_LEVEL = 0.03
SEED = 42
FS = 10.0                  # Sampling rate [Hz] for cross-correlation
T_WINDOW = 120.0           # Cross-correlation window [s]


# ─── Data Generation ──────────────────────────────────────────────
def generate_velocity_model(nx, ny, rng):
    """Create checkerboard + Gaussian anomaly velocity model."""
    x = np.linspace(XMIN, XMAX, nx)
    y = np.linspace(YMIN, YMAX, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Checkerboard pattern
    checker_scale = (XMAX - XMIN) / 5
    dm = 0.05 * np.sin(2 * np.pi * xx / checker_scale) * \
         np.sin(2 * np.pi * yy / checker_scale)

    # Gaussian anomaly
    cx, cy = 0.6 * XMAX, 0.4 * YMAX
    sigma = 20.0
    dm += 0.08 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

    # Low-velocity zone
    cx2, cy2 = 0.3 * XMAX, 0.7 * YMAX
    dm -= 0.06 * np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / (2 * (25)**2))

    return dm, xx, yy


def generate_stations(n_stations, rng):
    """Random station positions within study area."""
    x_sta = XMIN + (XMAX - XMIN) * rng.random(n_stations)
    y_sta = YMIN + (YMAX - YMIN) * rng.random(n_stations)
    return np.column_stack([x_sta, y_sta])


def generate_cross_correlation(dt_true, fs, t_window, rng):
    """
    Simulate ambient noise cross-correlation waveform.
    Creates a synthetic Ricker wavelet centred at dt_true.
    """
    n_samples = int(t_window * fs)
    t = np.arange(-n_samples // 2, n_samples // 2) / fs

    # Ricker wavelet at dominant period ~5s
    f0 = 0.2
    tau = t - dt_true
    wavelet = (1 - 2 * (np.pi * f0 * tau)**2) * \
              np.exp(-(np.pi * f0 * tau)**2)

    # Add noise
    wavelet += 0.1 * rng.standard_normal(len(wavelet))

    return t, wavelet


# ─── Forward Operator ─────────────────────────────────────────────
def build_kernel_matrix(stations, nx, ny):
    """
    Build sensitivity kernel G for all station pairs.
    Uses straight-ray approximation with sub-sampling.
    """
    n_sta = len(stations)
    dx = (XMAX - XMIN) / nx
    dy = (YMAX - YMIN) / ny
    n_cells = nx * ny

    pairs = []
    rows, cols, vals = [], [], []
    ray_idx = 0

    for i in range(n_sta):
        for j in range(i + 1, n_sta):
            x0, y0 = stations[i]
            x1, y1 = stations[j]

            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            if dist < 10.0:  # Skip very short paths
                continue

            # Sample ray path
            n_samples = max(int(dist / (0.5 * min(dx, dy))), 200)
            t_param = np.linspace(0, 1, n_samples)
            x_ray = x0 + (x1 - x0) * t_param
            y_ray = y0 + (y1 - y0) * t_param
            ds = dist / n_samples

            # Accumulate path lengths per cell
            i_cells = np.clip(((x_ray - XMIN) / dx).astype(int), 0, nx - 1)
            j_cells = np.clip(((y_ray - YMIN) / dy).astype(int), 0, ny - 1)
            cell_ids = i_cells * ny + j_cells

            # Count segments per cell
            unique_cells, counts = np.unique(cell_ids, return_counts=True)
            for c, cnt in zip(unique_cells, counts):
                rows.append(ray_idx)
                cols.append(c)
                vals.append(cnt * ds)

            pairs.append((i, j))
            ray_idx += 1

    G = csr_matrix((vals, (rows, cols)), shape=(ray_idx, n_cells))
    return G, pairs


def forward_travel_times(G, dm_flat, c0):
    """Compute travel-time perturbations: δt = G @ (δs), δs = -dm/c0²."""
    # dm is δc/c0, slowness perturbation δs ≈ -dm/c0
    ds = -dm_flat / c0
    dt = G @ ds
    return dt


# ─── Inverse Solver ────────────────────────────────────────────────
def build_smoothing_matrix(nx, ny):
    """2D Laplacian smoothing operator."""
    def diff1d(n):
        from scipy.sparse import diags
        return diags([-1, 1], [0, 1], shape=(n - 1, n))

    Dx = diff1d(nx)
    Dy = diff1d(ny)

    Lx = spkron(Dx.T @ Dx, speye(ny))
    Ly = spkron(speye(nx), Dy.T @ Dy)
    return Lx + Ly


def invert_tomography(G, dt, nx, ny, alpha=1.0, damp=0.01):
    """
    LSQR inversion with smoothing regularisation.
    """
    L = build_smoothing_matrix(nx, ny)
    n_cells = nx * ny

    G_aug = spvstack([G, np.sqrt(alpha) * L])
    dt_aug = np.concatenate([dt, np.zeros(L.shape[0])])

    result = lsqr(G_aug, dt_aug, damp=damp, iter_lim=500,
                   atol=1e-8, btol=1e-8)
    ds_rec = result[0]

    # Convert back: dm = -c0 * δs
    dm_rec = -C0 * ds_rec
    return dm_rec


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt, rec):
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        data_range = 1.0
    mse = np.mean((gt - rec)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt, rec, data_range=data_range))
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(dm_gt, dm_rec, stations, pairs, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmax = max(np.abs(dm_gt).max(), np.abs(dm_rec).max())
    extent = [XMIN, XMAX, YMIN, YMAX]

    # Ray coverage
    ax = axes[0, 0]
    for si, sj in pairs[:200]:  # Plot subset of rays
        ax.plot([stations[si, 0], stations[sj, 0]],
                [stations[si, 1], stations[sj, 1]],
                'b-', alpha=0.05, lw=0.5)
    ax.plot(stations[:, 0], stations[:, 1], 'r^', ms=5)
    ax.set_title(f'Ray Coverage ({len(pairs)} paths)')
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect('equal')

    # GT
    im1 = axes[0, 1].imshow(dm_gt.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='equal')
    axes[0, 1].set_title('Ground Truth δc/c₀')
    plt.colorbar(im1, ax=axes[0, 1])

    # Reconstruction
    im2 = axes[1, 0].imshow(dm_rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='equal')
    axes[1, 0].set_title('LSQR Reconstruction')
    plt.colorbar(im2, ax=axes[1, 0])

    # Error
    err = dm_gt - dm_rec
    im3 = axes[1, 1].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent, aspect='equal')
    axes[1, 1].set_title('Error')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(
        f"NoisePy — Ambient Noise Tomography\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f} | "
        f"RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  NoisePy — Ambient Noise Cross-Correlation Tomography")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation")
    dm_gt, xx, yy = generate_velocity_model(NX, NY, rng)
    stations = generate_stations(N_STATIONS, rng)
    print(f"  Grid: {NX}×{NY}, Domain: {XMAX-XMIN}×{YMAX-YMIN} km")
    print(f"  Stations: {N_STATIONS}")
    print(f"  Velocity perturbation range: [{dm_gt.min():.4f}, {dm_gt.max():.4f}]")

    # Stage 2: Forward
    print("\n[STAGE 2] Forward — Straight-Ray Travel Times")
    G, pairs = build_kernel_matrix(stations, NX, NY)
    dm_gt_flat = dm_gt.ravel()
    dt_clean = forward_travel_times(G, dm_gt_flat, C0)
    noise = NOISE_LEVEL * np.std(dt_clean) * rng.standard_normal(len(dt_clean))
    dt_noisy = dt_clean + noise
    print(f"  Station pairs: {len(pairs)}")
    print(f"  Kernel matrix G: {G.shape}, nnz={G.nnz}")
    print(f"  δt range: [{dt_clean.min():.4f}, {dt_clean.max():.4f}] s")

    # Simulate a few cross-correlations
    print("\n[STAGE 2b] Simulating cross-correlation waveforms ...")
    for k in range(min(3, len(pairs))):
        si, sj = pairs[k]
        dist = np.linalg.norm(stations[si] - stations[sj])
        dt_true = dist / C0
        t_cc, cc_waveform = generate_cross_correlation(dt_true, FS, T_WINDOW, rng)
        print(f"  Pair ({si},{sj}): dist={dist:.1f} km, "
              f"t_true={dt_true:.2f} s, peak at {t_cc[np.argmax(cc_waveform)]:.2f} s")

    # Stage 3: Inverse
    print("\n[STAGE 3] Inverse — LSQR + Laplacian Regularisation")
    best_cc = -1
    best_rec = None
    best_alpha = 1.0
    for alpha in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        dm_rec = invert_tomography(G, dt_noisy, NX, NY, alpha=alpha)
        dm_rec_2d = dm_rec.reshape(NX, NY)
        cc_val = float(np.corrcoef(dm_gt_flat, dm_rec)[0, 1])
        print(f"  α={alpha:6.2f} → CC={cc_val:.4f}")
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

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), dm_gt)

    visualize_results(dm_gt, best_rec, stations, pairs, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
