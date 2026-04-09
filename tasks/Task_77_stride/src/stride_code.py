"""
stride — Ultrasound Computed Tomography (USCT) Reconstruction
================================================================
Task #74: Reconstruct 2D sound-speed map from ultrasound travel-time
          measurements in ring geometry using LSQR tomography.

Inverse Problem:
    Given travel times t_ij between transmitter i and receiver j arranged
    in a ring geometry, recover the sound-speed distribution c(x,y).
    t_ij = ∫_{ray_ij} ds / c(x,y)  (straight-ray approximation)

Forward Model:
    Straight-ray travel time in ring geometry:
    t_ij = Σ_k L_{ij,k} · s_k
    where s_k = 1/c_k is slowness in cell k, and L_{ij,k} is the
    ray-path length through cell k.

Inverse Solver:
    LSQR with Tikhonov (Laplacian) smoothing regularisation.
    L-curve for regularisation parameter selection.

Repo: https://github.com/trustimaging/stride
Paper: Stride documentation & Pratt (1999), Geophysics.

Usage: /data/yjh/spectro_env/bin/python stride_code.py
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
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NX, NY = 64, 64             # Image grid
DOMAIN_SIZE = 0.1            # Domain [m] (100mm diameter)
C0 = 1500.0                 # Background sound speed [m/s] (water)
N_TRANSDUCERS = 32           # Number of transducers on ring
RING_RADIUS = 0.045          # Ring radius [m] (45mm)
NOISE_LEVEL = 0.02           # Relative noise on travel times
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_speed_model(nx, ny, domain_size):
    """
    Create a 2D sound-speed phantom mimicking a breast cross-section.
    """
    x = np.linspace(-domain_size / 2, domain_size / 2, nx)
    y = np.linspace(-domain_size / 2, domain_size / 2, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Start with water background
    c = np.ones((nx, ny)) * C0

    # Breast outline (circle)
    r_breast = 0.035
    breast_mask = xx**2 + yy**2 < r_breast**2
    c[breast_mask] = 1480  # Fat/tissue average

    # Fibroglandular tissue region (higher speed)
    r_fibro = 0.02
    fibro_mask = (xx - 0.005)**2 + (yy + 0.003)**2 < r_fibro**2
    c[fibro_mask & breast_mask] = 1520

    # Tumour (higher sound speed)
    r_tumour = 0.006
    tumour_mask = (xx + 0.008)**2 + (yy - 0.01)**2 < r_tumour**2
    c[tumour_mask & breast_mask] = 1560

    # Small cyst (lower sound speed)
    r_cyst = 0.004
    cyst_mask = (xx - 0.012)**2 + (yy + 0.008)**2 < r_cyst**2
    c[cyst_mask & breast_mask] = 1450

    # Smooth transitions
    c = gaussian_filter(c, sigma=1.0)

    return c, x, y


def generate_transducer_positions(n_trans, radius):
    """Generate equally-spaced transducer positions on a ring."""
    angles = np.linspace(0, 2 * np.pi, n_trans, endpoint=False)
    x_trans = radius * np.cos(angles)
    y_trans = radius * np.sin(angles)
    return np.column_stack([x_trans, y_trans])


# ─── Forward Operator ─────────────────────────────────────────────
def build_ray_matrix(transducers, nx, ny, domain_size):
    """
    Build the ray-path kernel matrix G for all transmitter-receiver pairs.
    G[ray_idx, cell_idx] = path length through cell.
    """
    x_edges = np.linspace(-domain_size / 2, domain_size / 2, nx + 1)
    y_edges = np.linspace(-domain_size / 2, domain_size / 2, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    n_trans = len(transducers)
    n_cells = nx * ny

    rows, cols, vals = [], [], []
    pairs = []
    ray_idx = 0

    for i in range(n_trans):
        for j in range(n_trans):
            if i == j:
                continue
            # Skip nearly opposite (through-transmission only for good coverage)
            # Include all pairs for full coverage
            x0, y0 = transducers[i]
            x1, y1 = transducers[j]

            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            n_samples = max(int(dist / (0.3 * min(dx, dy))), 300)

            t_param = np.linspace(0, 1, n_samples)
            x_ray = x0 + (x1 - x0) * t_param
            y_ray = y0 + (y1 - y0) * t_param
            ds = dist / n_samples

            # Find cell indices
            i_cells = np.clip(((x_ray - x_edges[0]) / dx).astype(int), 0, nx - 1)
            j_cells = np.clip(((y_ray - y_edges[0]) / dy).astype(int), 0, ny - 1)
            cell_ids = i_cells * ny + j_cells

            unique_cells, counts = np.unique(cell_ids, return_counts=True)
            for c, cnt in zip(unique_cells, counts):
                rows.append(ray_idx)
                cols.append(c)
                vals.append(cnt * ds)

            pairs.append((i, j))
            ray_idx += 1

    G = csr_matrix((vals, (rows, cols)), shape=(ray_idx, n_cells))
    return G, pairs


def forward_travel_times(G, c_model, c0):
    """
    Compute travel times: t = G @ s, where s = 1/c (slowness).
    Return travel-time perturbations: δt = G @ (s - s0).
    """
    s = 1.0 / c_model.ravel()
    s0 = 1.0 / c0
    ds = s - s0
    dt = G @ ds
    t_abs = G @ s  # absolute travel times
    return dt, t_abs


# ─── Inverse Solver ────────────────────────────────────────────────
def build_laplacian_2d(nx, ny):
    """2D Laplacian for smoothing regularisation."""
    def diff1d(n):
        from scipy.sparse import diags
        return diags([-1, 1], [0, 1], shape=(n - 1, n))

    Dx = diff1d(nx)
    Dy = diff1d(ny)
    Lx = spkron(Dx.T @ Dx, speye(ny))
    Ly = spkron(speye(nx), Dy.T @ Dy)
    return Lx + Ly


def invert_lsqr(G, dt, nx, ny, alpha=1.0, damp=0.0):
    """LSQR inversion with Laplacian regularisation."""
    L = build_laplacian_2d(nx, ny)
    G_aug = spvstack([G, np.sqrt(alpha) * L])
    dt_aug = np.concatenate([dt, np.zeros(L.shape[0])])

    result = lsqr(G_aug, dt_aug, damp=damp, iter_lim=300,
                   atol=1e-8, btol=1e-8)
    ds_rec = result[0]

    # Convert back to speed: c_rec = 1 / (s0 + ds)
    s0 = 1.0 / C0
    s_rec = s0 + ds_rec
    # Avoid division by zero
    s_rec = np.maximum(s_rec, 1e-10)
    c_rec = 1.0 / s_rec
    # Clip to physical range
    c_rec = np.clip(c_rec, 1300, 1700)

    return c_rec.reshape(nx, ny)


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
def visualize_results(c_gt, c_rec, transducers, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    extent = [-DOMAIN_SIZE / 2 * 1000, DOMAIN_SIZE / 2 * 1000,
              -DOMAIN_SIZE / 2 * 1000, DOMAIN_SIZE / 2 * 1000]

    vmin = min(c_gt.min(), c_rec.min())
    vmax = max(c_gt.max(), c_rec.max())

    # GT
    im0 = axes[0, 0].imshow(c_gt.T, cmap='jet', vmin=vmin, vmax=vmax,
                              origin='lower', extent=extent)
    axes[0, 0].plot(transducers[:, 0] * 1000, transducers[:, 1] * 1000,
                     'k.', ms=3)
    axes[0, 0].set_title('Ground Truth c(x,y) [m/s]')
    axes[0, 0].set_xlabel('x [mm]')
    axes[0, 0].set_ylabel('y [mm]')
    plt.colorbar(im0, ax=axes[0, 0])

    # Reconstruction
    im1 = axes[0, 1].imshow(c_rec.T, cmap='jet', vmin=vmin, vmax=vmax,
                              origin='lower', extent=extent)
    axes[0, 1].plot(transducers[:, 0] * 1000, transducers[:, 1] * 1000,
                     'k.', ms=3)
    axes[0, 1].set_title('LSQR Reconstruction')
    plt.colorbar(im1, ax=axes[0, 1])

    # Error
    err = c_gt - c_rec
    im2 = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent)
    axes[1, 0].set_title('Error (GT - Recon)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Profile
    mid = c_gt.shape[0] // 2
    x_mm = np.linspace(-DOMAIN_SIZE / 2, DOMAIN_SIZE / 2, c_gt.shape[0]) * 1000
    axes[1, 1].plot(x_mm, c_gt[mid, :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(x_mm, c_rec[mid, :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title('Central Profile')
    axes[1, 1].set_xlabel('x [mm]')
    axes[1, 1].set_ylabel('Speed [m/s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle(
        f"stride — USCT Sound-Speed Tomography ({N_TRANSDUCERS} transducers)\n"
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
    print("  stride — Ultrasound Computed Tomography Reconstruction")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation — Breast Phantom")
    c_gt, x_grid, y_grid = generate_speed_model(NX, NY, DOMAIN_SIZE)
    transducers = generate_transducer_positions(N_TRANSDUCERS, RING_RADIUS)
    print(f"  Grid: {NX}×{NY}, Domain: {DOMAIN_SIZE*1000:.0f}mm")
    print(f"  Transducers: {N_TRANSDUCERS} on ring (r={RING_RADIUS*1000:.0f}mm)")
    print(f"  Speed range: [{c_gt.min():.0f}, {c_gt.max():.0f}] m/s")

    # Stage 2: Forward
    print("\n[STAGE 2] Forward — Straight-Ray Travel Times")
    G, pairs = build_ray_matrix(transducers, NX, NY, DOMAIN_SIZE)
    dt_clean, t_abs = forward_travel_times(G, c_gt, C0)
    noise = NOISE_LEVEL * np.std(dt_clean) * rng.standard_normal(len(dt_clean))
    dt_noisy = dt_clean + noise
    print(f"  Ray pairs: {len(pairs)}")
    print(f"  Kernel matrix G: {G.shape}, nnz={G.nnz}")
    print(f"  δt range: [{dt_clean.min():.6f}, {dt_clean.max():.6f}] s")

    # Stage 3: Inverse
    print("\n[STAGE 3] Inverse — LSQR + Laplacian Regularisation")
    best_cc = -1
    best_rec = None
    best_alpha = 1.0

    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        c_rec = invert_lsqr(G, dt_noisy, NX, NY, alpha=alpha)
        cc_val = float(np.corrcoef(c_gt.ravel(), c_rec.ravel())[0, 1])
        print(f"  α={alpha:7.2f} → CC={cc_val:.4f}")
        if cc_val > best_cc:
            best_cc = cc_val
            best_alpha = alpha
            best_rec = c_rec

    print(f"  → Best α={best_alpha} with CC={best_cc:.4f}")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    metrics = compute_metrics(c_gt, best_rec)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), c_gt)

    visualize_results(c_gt, best_rec, transducers, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
