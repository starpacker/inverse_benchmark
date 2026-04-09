#!/usr/bin/env python3
"""
Task 136: Mueller Matrix Polarimetry via Dual Rotating Retarder
================================================================
Recovers a 4x4 Mueller matrix from simulated polarization intensity
measurements using the dual rotating retarder (DRR) technique.

Forward Model:
  I(theta_g, theta_a) = [1,0,0,0] @ M_PSA(theta_a) @ M_sample @ M_PSG(theta_g) @ [1,0,0,0]^T
  where M_PSG = M_retarder(theta_g) and M_PSA = M_analyzer @ M_retarder(theta_a)

Inverse: Least-squares via pseudo-inverse of the measurement matrix.
"""
import matplotlib
matplotlib.use('Agg')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, norm

# ── Reproducibility ──────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Output dirs ──────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  Mueller-matrix building blocks
# ═══════════════════════════════════════════════════════════════════

def mueller_rotation(theta):
    """Rotation matrix R(theta) for Mueller calculus (angle in radians)."""
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    return np.array([
        [1,  0,   0,   0],
        [0,  c2,  s2,  0],
        [0, -s2,  c2,  0],
        [0,  0,   0,   1],
    ])


def mueller_linear_retarder(delta, theta=0.0):
    """Mueller matrix of a linear retarder with retardance *delta* at angle *theta*."""
    cd = np.cos(delta)
    sd = np.sin(delta)
    M_ret = np.array([
        [1, 0,   0,    0],
        [0, 1,   0,    0],
        [0, 0,   cd,   sd],
        [0, 0,  -sd,   cd],
    ])
    R = mueller_rotation(theta)
    Rinv = mueller_rotation(-theta)
    return R @ M_ret @ Rinv


def mueller_linear_polarizer(theta=0.0, p=1.0):
    """Mueller matrix of a linear polarizer at angle *theta* with extinction ratio *p* (0..1)."""
    # p=1 → ideal polarizer; p=0 → blocks everything
    M_pol = 0.5 * np.array([
        [1 + p**2, 1 - p**2, 0, 0],
        [1 - p**2, 1 + p**2, 0, 0],
        [0,        0,       2*p, 0],
        [0,        0,       0,  2*p],
    ])
    # Wait - standard form for ideal polarizer (p=1 for transmittance, extinction = 0)
    # Let me use the standard form directly
    # For ideal polarizer at angle theta:
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    M = 0.5 * np.array([
        [1,    c2,     s2,    0],
        [c2,   c2**2,  c2*s2, 0],
        [s2,   c2*s2,  s2**2, 0],
        [0,    0,      0,     0],
    ])
    return M


def mueller_ideal_polarizer_h():
    """Ideal horizontal linear polarizer."""
    return 0.5 * np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=float)


def mueller_partial_polarizer(diattenuation=0.5, theta=0.0):
    """Partial polarizer with specified diattenuation D ∈ [0, 1]."""
    D = diattenuation
    # Transmittances
    q = 0.5 * (1 + D)  # major
    r = 0.5 * (1 - D)  # minor
    a = q + r
    b = q - r
    c = 2 * np.sqrt(q * r)
    M = np.array([
        [a, b, 0, 0],
        [b, a, 0, 0],
        [0, 0, c, 0],
        [0, 0, 0, c],
    ])
    if theta != 0.0:
        R = mueller_rotation(theta)
        Rinv = mueller_rotation(-theta)
        M = R @ M @ Rinv
    return M


def mueller_depolarizer(p=0.3):
    """Diagonal depolarizer (depolarization parameter p ∈ [0,1]; p=1 → no depolarization)."""
    return np.diag([1.0, p, p, p])


# ═══════════════════════════════════════════════════════════════════
#  Dual Rotating Retarder (DRR) Polarimeter
# ═══════════════════════════════════════════════════════════════════

def build_measurement_matrix(theta_g_list, theta_a_list, delta_g=np.pi/2, delta_a=np.pi/2):
    """
    Build the N×16 measurement (instrument) matrix **W** for a DRR polarimeter.

    The complete DRR system:
      PSG = Retarder(delta_g, theta_g) after a fixed horizontal polarizer (source)
      PSA = Fixed horizontal polarizer (detector) after Retarder(delta_a, theta_a)

    Intensity:
      I_k = first_row(M_PSA) @ M_sample @ first_col(M_PSG_full)
          = D^T @ M @ S = sum_{ij} D_i M_{ij} S_j
          = vec(M)^T  (S ⊗ D)    [using Kronecker/outer product]

    We need the PSG to include the source polarizer so that S varies with
    theta_g in all 4 Stokes components, not just 3.

    Parameters
    ----------
    theta_g_list : array-like  – generator retarder angles (rad)
    theta_a_list : array-like  – analyser retarder angles (rad)
    delta_g      : float       – generator retarder retardance (default QWP)
    delta_a      : float       – analyser retarder retardance (default QWP)

    Returns
    -------
    W : ndarray, shape (N, 16)
    """
    N = len(theta_g_list)
    W = np.zeros((N, 16))

    # Source: horizontally polarised light
    P_h = mueller_ideal_polarizer_h()
    S0 = np.array([1.0, 0.0, 0.0, 0.0])

    for k in range(N):
        # ── PSG side ──
        # Source polarizer (horizontal) → retarder at theta_g
        M_g = mueller_linear_retarder(delta_g, theta_g_list[k])
        # Full PSG Mueller: M_PSG = M_retarder_g @ M_polarizer_source
        M_psg = M_g @ P_h
        # Stokes vector generated = M_PSG @ S0 = first column of M_PSG (since S0 = [1,0,0,0])
        S_in = M_psg @ S0  # = M_psg[:, 0]

        # ── PSA side ──
        # Retarder at theta_a → detector polarizer (horizontal)
        M_a = mueller_linear_retarder(delta_a, theta_a_list[k])
        M_psa = P_h @ M_a
        # Detection vector = first row of M_PSA
        D_out = M_psa[0, :]

        # ── Measurement row ──
        # I_k = D_out^T M S_in = sum_{ij} D_out[i] * M[i,j] * S_in[j]
        # Vectorised: I_k = (D_out ⊗ S_in)^T  vec(M)
        # where vec(M) stacks M row-by-row → vec(M)[4*i+j] = M[i,j]
        # So W[k, 4*i+j] = D_out[i] * S_in[j]
        W[k, :] = np.outer(D_out, S_in).ravel()

    return W


def simulate_drr_measurements(M_sample, theta_g_list, theta_a_list,
                               delta_g=np.pi/2, delta_a=np.pi/2):
    """Simulate noiseless DRR intensity measurements for a given sample Mueller matrix."""
    W = build_measurement_matrix(theta_g_list, theta_a_list, delta_g, delta_a)
    m_vec = M_sample.ravel()  # 16-element vector (row-major)
    I = W @ m_vec
    return I, W


def recover_mueller(I_noisy, W):
    """Recover Mueller matrix from noisy intensity measurements via least squares."""
    m_vec, residuals, rank, sv = lstsq(W, I_noisy, rcond=None)
    M_recon = m_vec.reshape(4, 4)
    return M_recon


# ═══════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(M_true, M_recon):
    """Compute PSNR, RMSE, Frobenius error, and element-wise correlation."""
    diff = M_true - M_recon

    # Frobenius norm error
    frob_err = norm(diff, 'fro')

    # RMSE over all 16 elements
    rmse = np.sqrt(np.mean(diff**2))

    # PSNR (treating Mueller elements as "signal")
    max_val = np.max(np.abs(M_true))
    if max_val < 1e-15:
        max_val = 1.0
    if rmse < 1e-15:
        psnr = 100.0  # essentially perfect
    else:
        psnr = 20 * np.log10(max_val / rmse)

    # Element-wise Pearson correlation
    t = M_true.ravel()
    r = M_recon.ravel()
    if np.std(t) < 1e-15 or np.std(r) < 1e-15:
        cc = 1.0 if np.allclose(t, r) else 0.0
    else:
        cc = float(np.corrcoef(t, r)[0, 1])

    return {
        'PSNR_dB': round(float(psnr), 4),
        'RMSE': round(float(rmse), 8),
        'Frobenius_error': round(float(frob_err), 8),
        'CC': round(float(cc), 6),
    }


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_results(M_true, M_recon, metrics, save_path):
    """Generate comprehensive visualization of Mueller matrix recovery."""
    fig = plt.figure(figsize=(20, 16))

    # ── Panel 1: GT Mueller matrix as 4×4 heatmap ────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    vmax = max(np.max(np.abs(M_true)), np.max(np.abs(M_recon)))
    im1 = ax1.imshow(M_true, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax1.set_title('Ground Truth Mueller Matrix', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{M_true[i,j]:.3f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(M_true[i,j]) > 0.5*vmax else 'black')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # ── Panel 2: Reconstructed Mueller matrix ────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(M_recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax2.set_title('Reconstructed Mueller Matrix', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{M_recon[i,j]:.3f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(M_recon[i,j]) > 0.5*vmax else 'black')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # ── Panel 3: Error matrix |M_true − M_recon| ────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    err_mat = np.abs(M_true - M_recon)
    im3 = ax3.imshow(err_mat, cmap='hot', aspect='equal')
    ax3.set_title('Absolute Error |GT − Recon|', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{err_mat[i,j]:.4f}', ha='center', va='center', fontsize=8,
                     color='white' if err_mat[i,j] > 0.5*np.max(err_mat) else 'black')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # ── Panel 4: Element-wise bar comparison ─────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    idx = np.arange(16)
    labels = [f'M[{i},{j}]' for i in range(4) for j in range(4)]
    gt_vals = M_true.ravel()
    rc_vals = M_recon.ravel()
    width = 0.35
    ax4.bar(idx - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax4.bar(idx + width/2, rc_vals, width, label='Reconstructed', color='coral', alpha=0.8)
    ax4.set_xticks(idx)
    ax4.set_xticklabels(labels, rotation=65, ha='right', fontsize=7)
    ax4.set_ylabel('Element Value')
    ax4.set_title('Element-wise Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # ── Panel 5: Scatter plot GT vs Recon ────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(gt_vals, rc_vals, c='teal', s=60, edgecolors='k', linewidths=0.5, zorder=3)
    lims = [min(gt_vals.min(), rc_vals.min()) - 0.1, max(gt_vals.max(), rc_vals.max()) + 0.1]
    ax5.plot(lims, lims, 'k--', alpha=0.5, label='Ideal (y=x)')
    ax5.set_xlim(lims)
    ax5.set_ylim(lims)
    ax5.set_xlabel('Ground Truth')
    ax5.set_ylabel('Reconstructed')
    ax5.set_title(f'Correlation (CC = {metrics["CC"]:.4f})', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    ax5.set_aspect('equal')

    # ── Panel 6: Metrics summary ─────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    txt = (
        f"PSNR:  {metrics['PSNR_dB']:.2f} dB\n"
        f"RMSE:  {metrics['RMSE']:.6f}\n"
        f"Frobenius Error:  {metrics['Frobenius_error']:.6f}\n"
        f"Correlation (CC):  {metrics['CC']:.6f}\n"
        f"\n"
        f"Method: Dual Rotating Retarder\n"
        f"         Polarimetry (DRR)\n"
        f"Inverse: Least-Squares (pseudoinverse)\n"
        f"Sample: Partial polarizer + retarder"
    )
    ax6.text(0.1, 0.5, txt, transform=ax6.transAxes, fontsize=13,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('Recovery Metrics', fontsize=12, fontweight='bold')

    fig.suptitle('Mueller Matrix Recovery — Dual Rotating Retarder Polarimetry',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Task 136: Mueller Matrix Polarimetry (Dual Rotating Retarder)")
    print("=" * 65)

    # ── 1. Ground truth Mueller matrix ───────────────────────────
    #    Combination: partial polarizer (D=0.6, 30°) followed by
    #    quarter-wave retarder at 15°.  This gives a non-trivial
    #    Mueller matrix that exercises all 16 elements.
    M_pol = mueller_partial_polarizer(diattenuation=0.6, theta=np.deg2rad(30))
    M_ret = mueller_linear_retarder(delta=np.pi / 2, theta=np.deg2rad(15))
    M_true = M_ret @ M_pol
    # Normalise so M[0,0] = 1 (conventional)
    M_true = M_true / M_true[0, 0]
    print(f"\n[GT] Mueller matrix (4×4), M[0,0] = {M_true[0,0]:.4f}")
    print(M_true.round(6))

    # ── 2. DRR measurement angles ───────────────────────────────
    #    Classic 5:1 ratio for generator:analyser rotation.
    #    36 positions give 36 measurements (>16 needed).
    N_meas = 36
    theta_g = np.linspace(0, np.pi, N_meas, endpoint=False)      # generator
    theta_a = np.linspace(0, 5 * np.pi, N_meas, endpoint=False)  # analyser (5:1)

    # ── 3. Simulate measurements ─────────────────────────────────
    I_clean, W = simulate_drr_measurements(M_true, theta_g, theta_a)
    print(f"\n[SIM] {N_meas} clean intensity measurements generated.")
    print(f"      Measurement matrix W shape: {W.shape}, rank: {np.linalg.matrix_rank(W)}")

    # ── 4. Add Gaussian noise ────────────────────────────────────
    noise_sigma = 0.005 * np.max(np.abs(I_clean))
    noise = RNG.normal(0, noise_sigma, size=I_clean.shape)
    I_noisy = I_clean + noise
    snr = 20 * np.log10(norm(I_clean) / norm(noise)) if norm(noise) > 0 else 100
    print(f"[NOISE] sigma = {noise_sigma:.6f}, SNR = {snr:.1f} dB")

    # ── 5. Recover Mueller matrix ────────────────────────────────
    M_recon = recover_mueller(I_noisy, W)
    # Normalise
    if abs(M_recon[0, 0]) > 1e-10:
        M_recon = M_recon / M_recon[0, 0]
    print(f"\n[RECON] Recovered Mueller matrix:")
    print(M_recon.round(6))

    # ── 6. Compute metrics ───────────────────────────────────────
    metrics = compute_metrics(M_true, M_recon)
    print(f"\n[METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── 7. Save outputs ──────────────────────────────────────────
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), M_true)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), M_recon)
    print(f"\n[SAVE] ground_truth.npy  ({M_true.shape})")
    print(f"[SAVE] reconstruction.npy ({M_recon.shape})")

    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] metrics.json")

    # ── 8. Visualization ─────────────────────────────────────────
    plot_results(M_true, M_recon, metrics,
                 os.path.join(RESULTS_DIR, 'reconstruction_result.png'))

    print("\n" + "=" * 65)
    print("  DONE — all outputs in ./results/")
    print("=" * 65)


if __name__ == '__main__':
    main()
