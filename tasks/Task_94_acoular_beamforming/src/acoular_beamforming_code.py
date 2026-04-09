#!/usr/bin/env python3
"""
Acoustic Beamforming — Inverse Problem

Forward model: Source distribution q(x) → Cross-Spectral Matrix (CSM) C
    C = G @ diag(q) @ G^H + σ²I
    where G is the steering vector matrix (Green's function),
    g_i(x) = exp(-jk|x - m_i|) / (4π|x - m_i|)

Inverse problem: Given CSM C, recover source distribution q(x) on a focus grid.

Methods:
    1. Conventional Beamforming (Delay-and-Sum)
    2. CLEAN-SC deconvolution with Gaussian smoothing
    3. Direct CSM-based NNLS inversion

Metrics: PSNR, SSIM on the reconstructed source power map.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json
import os

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

FREQ = 8000.0
C_SOUND = 343.0
WAVELENGTH = C_SOUND / FREQ
K_WAVE = 2.0 * np.pi * FREQ / C_SOUND

N_MICS = 64
ARRAY_RADIUS = 0.5

Z_FOCUS = 1.5
GRID_SPAN = 0.8
GRID_RES = 31

SNR_DB = 30.0

CLEAN_ITERATIONS = 500
CLEAN_SAFETY = 0.5


# ============================================================
# Functions
# ============================================================

def create_spiral_array(n_mics, radius):
    """Archimedean spiral microphone array in z=0 plane."""
    angles = np.linspace(0, 4 * np.pi, n_mics, endpoint=False)
    radii = np.linspace(0.05, radius, n_mics)
    return np.column_stack([radii * np.cos(angles),
                            radii * np.sin(angles),
                            np.zeros(n_mics)])


def create_focus_grid(grid_span, grid_res, z_focus):
    """2D focus grid at distance z_focus."""
    coords = np.linspace(-grid_span / 2, grid_span / 2, grid_res)
    gx, gy = np.meshgrid(coords, coords)
    grid_points = np.column_stack([gx.ravel(), gy.ravel(),
                                    np.full(grid_res**2, z_focus)])
    return grid_points, coords


def compute_steering_vectors(mic_positions, grid_points, k):
    """Steering vector matrix G[i,j] = exp(-jkr) / (4πr)."""
    diff = mic_positions[:, np.newaxis, :] - grid_points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return np.exp(-1j * k * distances) / (4.0 * np.pi * distances)


def create_source_distribution(grid_points, grid_res):
    """3 Gaussian blob sources."""
    sources = [
        {'x': -0.12, 'y': 0.15,  'strength': 1.0},
        {'x': 0.18,  'y': -0.08, 'strength': 0.7},
        {'x': 0.0,   'y': -0.20, 'strength': 0.5},
    ]
    q = np.zeros(grid_res * grid_res)
    sigma = 0.04
    for s in sources:
        r2 = (grid_points[:, 0] - s['x'])**2 + (grid_points[:, 1] - s['y'])**2
        q += s['strength'] * np.exp(-r2 / (2 * sigma**2))
    return q


def forward_model(G, q, snr_db):
    """C = G diag(q) G^H + σ²I."""
    C_signal = G @ np.diag(q) @ G.conj().T
    sig_power = np.real(np.trace(C_signal)) / G.shape[0]
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    return C_signal + noise_power * np.eye(G.shape[0])


def conventional_beamforming(C, G):
    """Vectorized delay-and-sum beamforming."""
    CG = C @ G
    B = np.real(np.sum(G.conj() * CG, axis=0))
    norms = np.real(np.sum(G.conj() * G, axis=0))
    norms_sq = np.maximum(norms**2, 1e-30)
    return np.maximum(B / norms_sq, 0)


def clean_sc(C, G, n_iter=500, safety=0.5):
    """CLEAN-SC deconvolution."""
    n_grid = G.shape[1]
    C_rem = C.copy()
    q_clean = np.zeros(n_grid)
    g_norms_sq = np.real(np.sum(G.conj() * G, axis=0))

    for _ in range(n_iter):
        CG = C_rem @ G
        B = np.real(np.sum(G.conj() * CG, axis=0))
        B = np.maximum(B / np.maximum(g_norms_sq**2, 1e-30), 0)

        if np.max(B) < 1e-15:
            break

        j = np.argmax(B)
        g_j = G[:, j:j+1]
        gns = g_norms_sq[j]
        if gns < 1e-30:
            break

        Cg = C_rem @ g_j
        gCg = np.real(g_j.conj().T @ Cg)[0, 0]
        if gCg < 1e-30:
            break

        h = Cg / gCg * gns
        strength = safety * B[j]
        q_clean[j] += strength

        C_rem -= strength * (h @ h.conj().T) / (gns**2)
        C_rem = 0.5 * (C_rem + C_rem.conj().T)

    return q_clean


def csm_nnls_inversion(C, G, alpha=1e-2):
    """
    Direct CSM-based NNLS inversion.
    
    Vectorize the CSM: c = vec(C) where we use the upper triangular part.
    Forward: c_ij = sum_k q_k * g_ik * conj(g_jk)
    Build matrix A where A[(i,j), k] = Re/Im parts of g_ik * conj(g_jk)
    Solve with NNLS.
    """
    n_mics = G.shape[0]
    n_grid = G.shape[1]

    # Use diagonal elements only (auto-spectra) for speed and stability
    # C_ii = sum_k |g_ik|^2 * q_k + noise
    # This is a simpler linear system: A @ q = c_diag
    A_diag = np.abs(G)**2  # (n_mics, n_grid)

    # Remove diagonal noise estimate
    C_diag = np.real(np.diag(C))

    # Also use real parts of off-diagonal (cross-spectra)
    # For selected pairs to keep system manageable
    pairs = []
    vals = []
    A_rows = []

    # Diagonal elements
    for i in range(n_mics):
        A_rows.append(A_diag[i, :])
        vals.append(C_diag[i])

    # Off-diagonal: use real parts of upper triangle (every other pair for speed)
    step = max(1, n_mics // 16)
    for i in range(0, n_mics, step):
        for j in range(i + 1, n_mics, step):
            row = np.real(G[i, :] * G[j, :].conj())
            A_rows.append(row)
            vals.append(np.real(C[i, j]))

    A_mat = np.array(A_rows)
    b_vec = np.array(vals)

    # Tikhonov regularization
    n_rows = A_mat.shape[0]
    A_reg = np.vstack([A_mat, np.sqrt(alpha) * np.eye(n_grid)])
    b_reg = np.concatenate([b_vec, np.zeros(n_grid)])

    q_sol, _ = nnls(A_reg, b_reg)
    return q_sol


def compute_metrics_linear(gt_map_2d, recon_map_2d):
    """Compute PSNR/SSIM on normalized linear-scale maps."""
    gt_max = np.max(gt_map_2d)
    if gt_max <= 0:
        gt_max = 1.0

    gt_n = gt_map_2d / gt_max
    recon_max = np.max(recon_map_2d)
    if recon_max <= 0:
        recon_n = np.zeros_like(recon_map_2d)
    else:
        recon_n = recon_map_2d / recon_max

    # Scale recon to minimize MSE (optimal scaling)
    scale = np.sum(gt_n * recon_n) / (np.sum(recon_n**2) + 1e-30)
    recon_scaled = np.clip(recon_n * scale, 0, 1)

    psnr = peak_signal_noise_ratio(gt_n, recon_scaled, data_range=1.0)
    ssim = structural_similarity(gt_n, recon_scaled, data_range=1.0)
    return psnr, ssim


# Remove CSM diagonal (noise removal technique)
def remove_csm_diagonal(C):
    """Set CSM diagonal to zero to remove uncorrelated noise."""
    C_clean = C.copy()
    np.fill_diagonal(C_clean, 0)
    return C_clean


def to_db(source_map, dynamic_range=30.0):
    """Convert to dB with dynamic range."""
    mx = np.max(source_map)
    if mx <= 0:
        return np.full_like(source_map, -dynamic_range)
    n = np.maximum(source_map / mx, 10 ** (-dynamic_range / 10))
    return 10.0 * np.log10(n)


def plot_results(coords, gt_2d, bf_2d, clean_2d, nnls_2d,
                 mic_pos, metrics_dict, save_path):
    """4-panel plot."""
    extent = [coords[0], coords[-1], coords[0], coords[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    panels = [
        (axes[0, 0], gt_2d, 'Ground Truth Source Distribution', None),
        (axes[0, 1], bf_2d, 'Conventional Beamforming',
         f"PSNR={metrics_dict['conv']['psnr']:.2f}dB, SSIM={metrics_dict['conv']['ssim']:.4f}"),
        (axes[1, 0], clean_2d, 'CLEAN-SC Deconvolution',
         f"PSNR={metrics_dict['clean']['psnr']:.2f}dB, SSIM={metrics_dict['clean']['ssim']:.4f}"),
        (axes[1, 1], nnls_2d, 'NNLS Inversion',
         f"PSNR={metrics_dict['nnls']['psnr']:.2f}dB, SSIM={metrics_dict['nnls']['ssim']:.4f}"),
    ]

    for ax, data, title, subtitle in panels:
        db = to_db(data)
        im = ax.imshow(db, extent=extent, origin='lower', cmap='hot',
                        vmin=-30, vmax=0, aspect='equal')
        full_title = title if subtitle is None else f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=12, fontweight='bold' if subtitle is None else 'normal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.colorbar(im, ax=ax, label='Power [dB]')
        ax.scatter(mic_pos[:, 0], mic_pos[:, 1],
                   c='cyan', s=8, alpha=0.5, marker='.', zorder=5)

    plt.suptitle(f'Acoustic Beamforming: Source Localization\n'
                 f'({N_MICS} mics, f={FREQ:.0f}Hz, λ={WAVELENGTH*100:.1f}cm, SNR={SNR_DB:.0f}dB)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Acoustic Beamforming — Inverse Problem")
    print("=" * 60)

    # Setup
    print("\n[1/7] Creating spiral microphone array...")
    mic_pos = create_spiral_array(N_MICS, ARRAY_RADIUS)

    print("[2/7] Creating focus grid...")
    grid_pts, coords = create_focus_grid(GRID_SPAN, GRID_RES, Z_FOCUS)
    n_grid = grid_pts.shape[0]
    print(f"  Grid: {GRID_RES}x{GRID_RES} = {n_grid}")

    print("[3/7] Creating source distribution...")
    q_gt = create_source_distribution(grid_pts, GRID_RES)

    print("[4/7] Computing steering vectors...")
    G = compute_steering_vectors(mic_pos, grid_pts, K_WAVE)

    print("[5/7] Forward model → CSM...")
    C = forward_model(G, q_gt, SNR_DB)
    print(f"  Hermitian: {np.allclose(C, C.conj().T)}")

    # Remove diagonal for noise-free beamforming
    C_clean = remove_csm_diagonal(C)

    # Inverse
    print("[6/7] Solving inverse problem...")

    # Conventional
    print("  [6a] Conventional beamforming...")
    B_conv = conventional_beamforming(C_clean, G)
    psnr_bf, ssim_bf = compute_metrics_linear(
        q_gt.reshape(GRID_RES, GRID_RES), B_conv.reshape(GRID_RES, GRID_RES))
    print(f"       PSNR={psnr_bf:.2f}dB, SSIM={ssim_bf:.4f}")

    # CLEAN-SC + smooth
    print("  [6b] CLEAN-SC...")
    q_clean_raw = clean_sc(C_clean, G, n_iter=CLEAN_ITERATIONS, safety=CLEAN_SAFETY)
    q_clean = gaussian_filter(q_clean_raw.reshape(GRID_RES, GRID_RES), sigma=1.5).ravel()
    psnr_cl, ssim_cl = compute_metrics_linear(
        q_gt.reshape(GRID_RES, GRID_RES), q_clean.reshape(GRID_RES, GRID_RES))
    print(f"       PSNR={psnr_cl:.2f}dB, SSIM={ssim_cl:.4f}")

    # NNLS
    print("  [6c] CSM-based NNLS inversion...")
    q_nnls_raw = csm_nnls_inversion(C_clean, G, alpha=1e-2)
    q_nnls = gaussian_filter(q_nnls_raw.reshape(GRID_RES, GRID_RES), sigma=1.0).ravel()
    psnr_nn, ssim_nn = compute_metrics_linear(
        q_gt.reshape(GRID_RES, GRID_RES), q_nnls.reshape(GRID_RES, GRID_RES))
    print(f"       PSNR={psnr_nn:.2f}dB, SSIM={ssim_nn:.4f}")

    # Best
    results = {
        'conventional': {'psnr': psnr_bf, 'ssim': ssim_bf, 'map': B_conv},
        'clean_sc': {'psnr': psnr_cl, 'ssim': ssim_cl, 'map': q_clean},
        'nnls': {'psnr': psnr_nn, 'ssim': ssim_nn, 'map': q_nnls},
    }
    best_name = max(results, key=lambda m: results[m]['psnr'])
    best = results[best_name]
    print(f"\n  Best: {best_name} (PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f})")

    # Save
    print("[7/7] Saving results...")
    gt_2d = q_gt.reshape(GRID_RES, GRID_RES)
    recon_2d = best['map'].reshape(GRID_RES, GRID_RES)

    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), gt_2d)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), recon_2d)

    metrics = {
        'psnr_db': round(best['psnr'], 2),
        'ssim': round(best['ssim'], 4),
        'best_method': best_name,
        'conventional': {'psnr_db': round(psnr_bf, 2), 'ssim': round(ssim_bf, 4)},
        'clean_sc': {'psnr_db': round(psnr_cl, 2), 'ssim': round(ssim_cl, 4)},
        'nnls': {'psnr_db': round(psnr_nn, 2), 'ssim': round(ssim_nn, 4)},
        'parameters': {
            'frequency_hz': FREQ, 'n_mics': N_MICS,
            'grid_resolution': GRID_RES, 'snr_db': SNR_DB, 'z_focus_m': Z_FOCUS,
        }
    }
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_results(coords, gt_2d,
                 B_conv.reshape(GRID_RES, GRID_RES),
                 q_clean.reshape(GRID_RES, GRID_RES),
                 q_nnls.reshape(GRID_RES, GRID_RES),
                 mic_pos,
                 {'conv': {'psnr': psnr_bf, 'ssim': ssim_bf},
                  'clean': {'psnr': psnr_cl, 'ssim': ssim_cl},
                  'nnls': {'psnr': psnr_nn, 'ssim': ssim_nn}},
                 os.path.join(RESULTS_DIR, 'reconstruction_result.png'))

    print(f"\n{'=' * 60}")
    print(f"FINAL ({best_name}): PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}")
    print(f"{'=' * 60}")
    return metrics


if __name__ == '__main__':
    main()
