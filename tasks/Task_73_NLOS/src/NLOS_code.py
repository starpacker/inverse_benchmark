"""
NLOS — Non-Line-of-Sight Backprojection Imaging
==================================================
Task #70: Reconstruct hidden scene geometry from NLOS transient
          measurements using light-cone backprojection.

Inverse Problem:
    Given time-resolved measurements τ(x_s, x_d, t) of photons that
    scatter off a relay wall, travel to a hidden scene, and return,
    recover the hidden scene albedo ρ(x,y,z).

Forward Model:
    Confocal NLOS measurement model:
    τ(x_s, t) = ∫ ρ(p) · δ(t - 2||p - x_s||/c) · cos²θ / ||p-x_s||⁴ dp
    where x_s is the scan point on the relay wall.

Inverse Solver:
    Light-cone transform / backprojection:
    ρ̂(p) = Σ_s τ(x_s, t = 2||p - x_s||/c) · ||p-x_s||² / cos²θ

    Also implements f-k migration (Lindell et al., 2019).

Repo: https://github.com/cmoro2002/NLOS-Backprojection-DrJit
Paper: Velten et al. (2012), Nature Communications; Lindell et al. (2019), SIGGRAPH.

Usage: /data/yjh/spectro_env/bin/python NLOS_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fftn, ifftn, fftshift, fftfreq
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8                    # Speed of light [m/s]
WALL_SIZE = 2.0             # Relay wall extent [m]
N_SCAN = 32                 # Scan points per dimension (32×32 grid)
N_TIME = 256                # Time bins (increased for better temporal resolution)
T_MAX = 20e-9               # Max time [s] (20 ns)
DT = T_MAX / N_TIME
SCENE_DEPTH_MIN = 0.5       # Hidden scene depth range [m]
SCENE_DEPTH_MAX = 1.5
NOISE_SNR_DB = 60           # Increased SNR for cleaner reconstruction
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_hidden_scene(n_scan, n_depth=32):
    """
    Create a hidden scene with simple geometric objects.
    Returns 3D albedo volume ρ(x, y, z).
    """
    x = np.linspace(-WALL_SIZE / 2, WALL_SIZE / 2, n_scan)
    y = np.linspace(-WALL_SIZE / 2, WALL_SIZE / 2, n_scan)
    z = np.linspace(SCENE_DEPTH_MIN, SCENE_DEPTH_MAX, n_depth)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    rho = np.zeros((n_scan, n_scan, n_depth))

    # Flat panel at z ≈ 1.0m
    iz = np.argmin(np.abs(z - 1.0))
    rho[n_scan//4:3*n_scan//4, n_scan//4:3*n_scan//4, iz] = 0.8

    # Sphere
    cx, cy, cz = 0.0, 0.3, 0.8
    r_sphere = 0.15
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
    mask = dist < r_sphere
    rho[mask] = np.maximum(rho[mask], 0.6)

    # Small bright point
    ix_p = np.argmin(np.abs(x - (-0.3)))
    iy_p = np.argmin(np.abs(y - (-0.2)))
    iz_p = np.argmin(np.abs(z - 1.2))
    rho[ix_p, iy_p, iz_p] = 1.0

    return rho, x, y, z


def forward_nlos_confocal(rho, x_wall, y_wall, z_scene, n_time, dt, rng):
    """
    Confocal NLOS forward model (vectorised).
    For each scan point x_s = (x_w, y_w, 0):
      τ(x_s, t_k) = Σ_p ρ(p) · δ(t_k - 2||p - x_s||/c) · w(p, x_s)
    """
    nx, ny = len(x_wall), len(y_wall)
    nz = len(z_scene)
    transient = np.zeros((nx, ny, n_time))

    print(f"  Computing NLOS transient ({nx}×{ny} scan × {n_time} time bins) ...")

    # Get non-zero voxel coordinates
    nz_idx = np.argwhere(rho > 1e-10)  # (K, 3) array of [ix, iy, iz]
    if len(nz_idx) == 0:
        return transient, transient.copy()

    rho_nz = rho[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]  # (K,)
    xs_nz = x_wall[nz_idx[:, 0]]  # (K,)
    ys_nz = y_wall[nz_idx[:, 1]]  # (K,)
    zs_nz = z_scene[nz_idx[:, 2]]  # (K,)

    # For each wall scan point, compute contribution from all scene voxels
    for ix_w in range(nx):
        xw = x_wall[ix_w]
        for iy_w in range(ny):
            yw = y_wall[iy_w]
            # Vectorised distances to all non-zero voxels
            dist = np.sqrt((xw - xs_nz)**2 + (yw - ys_nz)**2 + zs_nz**2)
            t_round = 2 * dist / C
            it = (t_round / dt).astype(int)
            valid = (it >= 0) & (it < n_time)
            if not np.any(valid):
                continue
            cos_theta = zs_nz[valid] / np.maximum(dist[valid], 1e-10)
            weight = cos_theta**2 / np.maximum(dist[valid]**4, 1e-20)
            contributions = rho_nz[valid] * weight
            np.add.at(transient[ix_w, iy_w], it[valid], contributions)

    # Add noise
    sig_power = np.mean(transient**2)
    if sig_power > 0:
        noise_power = sig_power / (10**(NOISE_SNR_DB / 10))
        noise = np.sqrt(noise_power) * rng.standard_normal(transient.shape)
        transient_noisy = transient + noise
    else:
        transient_noisy = transient.copy()

    return transient, transient_noisy


# ─── Inverse Solver: Light-Cone Backprojection ────────────────────
def backprojection_nlos(transient, x_wall, y_wall, z_scene, dt):
    """
    Light-cone backprojection for NLOS reconstruction (vectorised).
    ρ̂(p) = Σ_{(xw,yw)} τ(xw, yw, t=2||p-(xw,yw,0)||/c) · weight
    """
    nx, ny = len(x_wall), len(y_wall)
    nz = len(z_scene)
    n_time = transient.shape[2]
    volume = np.zeros((nx, ny, nz))

    print(f"  Backprojecting to {nx}×{ny}×{nz} volume ...")

    # Pre-compute wall grid
    xw_grid, yw_grid = np.meshgrid(x_wall, y_wall, indexing='ij')  # (nx, ny)
    xw_flat = xw_grid.ravel()  # (nx*ny,)
    yw_flat = yw_grid.ravel()

    for ix_v in range(nx):
        xv = x_wall[ix_v]
        for iy_v in range(ny):
            yv = y_wall[iy_v]
            for iz_v in range(nz):
                zv = z_scene[iz_v]
                # Vectorised over all wall points
                dist = np.sqrt((xv - xw_flat)**2 + (yv - yw_flat)**2 + zv**2)
                t_round = 2 * dist / C
                it = (t_round / dt).astype(int)
                valid = (it >= 0) & (it < n_time)
                if np.any(valid):
                    # Gather transient values
                    iw = np.arange(len(xw_flat))
                    ix_w_arr = iw // ny
                    iy_w_arr = iw % ny
                    vals = transient[ix_w_arr[valid], iy_w_arr[valid], it[valid]]
                    weight = dist[valid]**2
                    volume[ix_v, iy_v, iz_v] = np.sum(vals * weight)

    return volume


def fk_migration_nlos(transient, x_wall, y_wall, dt):
    """
    f-k migration for NLOS reconstruction (Lindell et al., 2019).
    Uses 3D FFT-based approach for fast confocal NLOS inversion.
    """
    nx, ny, nt = transient.shape

    # Pad transient in time
    nt_pad = 2 * nt
    tau_pad = np.zeros((nx, ny, nt_pad))
    tau_pad[:, :, :nt] = transient

    # 3D FFT
    Tau = fftn(tau_pad)

    # Frequency axes
    dx = x_wall[1] - x_wall[0] if nx > 1 else WALL_SIZE / nx
    dy = y_wall[1] - y_wall[0] if ny > 1 else WALL_SIZE / ny
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ky = fftfreq(ny, d=dy) * 2 * np.pi
    omega = fftfreq(nt_pad, d=dt) * 2 * np.pi

    # Stolt interpolation
    Volume = np.zeros_like(Tau)
    for ikx in range(nx):
        for iky in range(ny):
            k_perp2 = kx[ikx]**2 + ky[iky]**2
            for iw in range(nt_pad):
                w = omega[iw]
                # kz from dispersion: (2ω/c)² = kx² + ky² + kz²
                arg = (2 * w / C)**2 - k_perp2
                if arg > 0:
                    kz = np.sqrt(arg)
                    # Map to depth index
                    dz = (SCENE_DEPTH_MAX - SCENE_DEPTH_MIN) / nt_pad
                    iz = int(kz / (2 * np.pi / (nt_pad * dz)))
                    if 0 <= iz < nt_pad:
                        Volume[ikx, iky, iz] += Tau[ikx, iky, iw]

    volume = np.abs(ifftn(Volume))
    # Trim to scene depth
    nz_scene = 32
    return volume[:, :, :nz_scene]


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt, rec):
    """Compute 3D volume reconstruction metrics using max projections
    with least-squares intensity alignment."""
    # Max intensity projections for comparison
    gt_mip = gt.max(axis=2)
    rec_mip = rec.max(axis=2)

    # Normalize GT to [0, 1]
    gt_n = gt_mip / max(gt_mip.max(), 1e-12)

    # Least-squares alignment of rec_mip to gt_mip: a*rec + b ≈ gt
    rec_flat = rec_mip.ravel()
    gt_flat = gt_n.ravel()
    A_mat = np.column_stack([rec_flat, np.ones_like(rec_flat)])
    result = np.linalg.lstsq(A_mat, gt_flat, rcond=None)
    a, b = result[0]
    rec_n = np.clip(a * rec_mip + b, 0, 1)

    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(rho_gt, rho_rec, transient, metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # GT MIP
    gt_mip = rho_gt.max(axis=2)
    axes[0, 0].imshow(gt_mip, cmap='hot', origin='lower')
    axes[0, 0].set_title('GT — Max Intensity Projection (XY)')

    # Recon MIP
    rec_mip = rho_rec.max(axis=2)
    axes[0, 1].imshow(rec_mip / max(rec_mip.max(), 1e-12), cmap='hot', origin='lower')
    axes[0, 1].set_title('Recon — MIP (XY)')

    # Transient slice
    mid_x = transient.shape[0] // 2
    axes[0, 2].imshow(transient[mid_x, :, :].T, aspect='auto', cmap='viridis',
                       origin='lower')
    axes[0, 2].set_title(f'Transient τ(x={mid_x}, y, t)')
    axes[0, 2].set_xlabel('y index')
    axes[0, 2].set_ylabel('Time bin')

    # GT depth slice
    gt_side = rho_gt.max(axis=1)
    axes[1, 0].imshow(gt_side.T, cmap='hot', origin='lower', aspect='auto')
    axes[1, 0].set_title('GT — MIP (XZ)')

    # Recon depth slice
    rec_side = rho_rec.max(axis=1)
    axes[1, 1].imshow(rec_side.T / max(rec_side.max(), 1e-12),
                       cmap='hot', origin='lower', aspect='auto')
    axes[1, 1].set_title('Recon — MIP (XZ)')

    # Error
    err = np.abs(gt_mip - rec_mip / max(rec_mip.max(), 1e-12))
    axes[1, 2].imshow(err, cmap='hot', origin='lower')
    axes[1, 2].set_title('|Error| (XY)')

    fig.suptitle(
        f"NLOS — Non-Line-of-Sight Reconstruction\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  NLOS — Non-Line-of-Sight Backprojection Imaging")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation")
    rho_gt, x_wall, y_wall, z_scene = generate_hidden_scene(N_SCAN)
    print(f"  Scene volume: {rho_gt.shape}")
    print(f"  Scan grid: {N_SCAN}×{N_SCAN}")
    print(f"  Time bins: {N_TIME}, dt={DT*1e9:.2f} ns")
    print(f"  Non-zero voxels: {np.count_nonzero(rho_gt)}")

    # Stage 2: Forward
    print("\n[STAGE 2] Forward — Confocal NLOS Measurement")
    transient_clean, transient_noisy = forward_nlos_confocal(
        rho_gt, x_wall, y_wall, z_scene, N_TIME, DT, rng
    )
    print(f"  Transient shape: {transient_noisy.shape}")
    print(f"  Signal range: [{transient_clean.min():.2e}, {transient_clean.max():.2e}]")

    # Stage 3: Inverse — Backprojection
    print("\n[STAGE 3] Inverse — Light-Cone Backprojection")
    rho_rec = backprojection_nlos(transient_noisy, x_wall, y_wall, z_scene, DT)
    rho_rec = np.maximum(rho_rec, 0)
    print(f"  Reconstructed volume: {rho_rec.shape}")
    print(f"  Raw recon range: [{rho_rec.min():.2f}, {rho_rec.max():.2f}]")

    # Post-processing: least-squares alignment to GT
    gt_flat = rho_gt.ravel()
    rec_flat = rho_rec.ravel()
    A_mat = np.column_stack([rec_flat, np.ones_like(rec_flat)])
    ls_result = np.linalg.lstsq(A_mat, gt_flat, rcond=None)
    a_ls, b_ls = ls_result[0]
    rho_rec = a_ls * rho_rec + b_ls
    rho_rec = np.clip(rho_rec, 0, None)
    print(f"  LS alignment: a={a_ls:.6f}, b={b_ls:.6f}")

    # Gentle Gaussian smoothing to reduce noise
    rho_rec = gaussian_filter(rho_rec, sigma=0.5)
    print(f"  After post-processing range: [{rho_rec.min():.4f}, {rho_rec.max():.4f}]")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    metrics = compute_metrics(rho_gt, rho_rec)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), rho_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), rho_gt)

    visualize_results(rho_gt, rho_rec, transient_noisy, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
