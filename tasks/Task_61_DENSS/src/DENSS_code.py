"""
DENSS — Small-Angle X-ray Scattering (SAXS) Density Reconstruction
====================================================================
Task: Recover 3D electron density from 1D SAXS intensity profile.

Inverse Problem:
    Given I(q) = |F(q)|² (radially averaged scattering intensity),
    recover the 3D electron density ρ(r) of a macromolecule.

Forward Model:
    Debye formula: I(q) = Σᵢ Σⱼ fᵢ fⱼ sin(q·rᵢⱼ)/(q·rᵢⱼ)
    or equivalently the spherical average of |FT{ρ(r)}|².

Inverse Solver:
    Iterative structure factor retrieval (DENSS algorithm):
    alternating real-space (support/positivity) and reciprocal-space
    (intensity matching) constraints.

Repo: https://github.com/tdgrant1/denss
Paper: Grant (2018), Nature Methods, 15, 191–193.

Usage:
    /data/yjh/spectro_env/bin/python DENSS_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
from skimage.metrics import structural_similarity as ssim_fn

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_GRID = 32         # 3D grid size
VOXEL_SIZE = 2.0    # Å per voxel
Q_MAX = 0.5         # Å^-1
N_Q = 100           # number of q bins
NOISE_PCT = 0.001   # 0.1% noise on I(q)
SEED = 42

# DENSS-like parameters
N_ITER = 10000
OVERSAMPLING = 2     # oversampling ratio
N_RUNS = 5           # number of independent runs to average


# ═══════════════════════════════════════════════════════════
# 2. Ground Truth: Synthetic Density (Two-Domain Protein)
# ═══════════════════════════════════════════════════════════
def create_gt_density(N):
    """Create a synthetic 3D electron density resembling a small protein."""
    rng = np.random.default_rng(SEED)
    density = np.zeros((N, N, N))
    c = N // 2

    # Main body (ellipsoidal)
    z, y, x = np.mgrid[:N, :N, :N] - c
    ellipsoid = (x/5)**2 + (y/4)**2 + (z/6)**2
    density[ellipsoid < 1] = 1.0

    # Secondary domain (smaller sphere offset)
    sphere2 = (x - 4)**2 + (y + 3)**2 + (z - 2)**2
    density[sphere2 < 9] = 0.8

    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=1.0)

    # Normalise
    density = density / density.max()
    return density


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator (3D FFT → Radial Average)
# ═══════════════════════════════════════════════════════════
def forward_operator(density, voxel_size, n_q, q_max):
    """
    Compute 1D SAXS profile I(q) from 3D electron density.

    I(q) = spherical_average( |FFT{ρ(r)}|² )

    Parameters
    ----------
    density : 3D array  Electron density.
    voxel_size : float   Voxel size [Å].
    n_q : int            Number of q bins.
    q_max : float        Maximum q [Å^-1].

    Returns
    -------
    q_bins : array  q values [Å^-1].
    I_q : array     Scattering intensity.
    """
    N = density.shape[0]

    # 3D FFT
    F = fftshift(fftn(ifftshift(density)))
    I_3d = np.abs(F) ** 2

    # q-grid
    freq = fftfreq(N, d=voxel_size)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx**2 + qy**2 + qz**2)

    # Radial average (spherical shells)
    q_bins = np.linspace(0.01, q_max, n_q)
    dq = q_bins[1] - q_bins[0]
    I_q = np.zeros(n_q)

    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq/2) & (q_3d < qc + dq/2)
        if mask.sum() > 0:
            I_q[i] = np.mean(I_3d[mask])

    # Normalise
    I_q = I_q / I_q.max()
    return q_bins, I_q


# ═══════════════════════════════════════════════════════════
# 4. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic SAXS data from GT density."""
    print("[DATA] Creating synthetic 3D electron density ...")
    density_gt = create_gt_density(N_GRID)
    print(f"[DATA] Density shape: {density_gt.shape}, "
          f"range [{density_gt.min():.3f}, {density_gt.max():.3f}]")

    print("[DATA] Computing SAXS intensity profile ...")
    q, I_clean = forward_operator(density_gt, VOXEL_SIZE, N_Q, Q_MAX)

    rng = np.random.default_rng(SEED)
    I_noisy = I_clean * (1 + NOISE_PCT * rng.standard_normal(N_Q))
    I_noisy = np.maximum(I_noisy, 0)

    print(f"[DATA] I(q) range: [{I_clean.min():.3e}, {I_clean.max():.3e}]")
    return q, I_noisy, I_clean, density_gt


# ═══════════════════════════════════════════════════════════
# 5. Inverse Solver (DENSS-like Iterative Phasing)
# ═══════════════════════════════════════════════════════════
def reconstruct(q_data, I_data, density_gt_shape):
    """
    SAXS density reconstruction via gradient-based optimisation.

    Instead of iterative phase retrieval (which is fundamentally limited
    for the 1D → 3D case), we directly minimise:

        L(ρ) = Σ_i [ I_model(q_i) - I_data(q_i) ]²

    where I_model is the radially-averaged power spectrum of ρ,
    subject to:
      - positivity: ρ ≥ 0
      - compact support (via shrink-wrap)
      - smoothness (via gradient penalty)

    The gradient ∂L/∂ρ is computed via the adjoint of the forward model
    (FFT-based), enabling efficient updates.

    Parameters
    ----------
    q_data : array      q values.
    I_data : array      Measured I(q).
    density_gt_shape : tuple  Expected output shape.

    Returns
    -------
    density_rec : 3D array  Reconstructed density.
    """
    from scipy.ndimage import gaussian_filter

    print(f"[RECON] Gradient-based SAXS reconstruction ({N_ITER} iterations) ...")
    N = density_gt_shape[0]

    # ── q-grid for radial binning ──
    freq = fftfreq(N, d=VOXEL_SIZE)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx**2 + qy**2 + qz**2)

    # Radial bin assignments (same bins as forward operator)
    q_bins = np.linspace(0.01, Q_MAX, N_Q)
    dq = q_bins[1] - q_bins[0]
    bin_index = np.full(q_3d.shape, -1, dtype=int)
    bin_count = np.zeros(N_Q)
    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq / 2) & (q_3d < qc + dq / 2)
        bin_index[mask] = i
        bin_count[i] = mask.sum()

    # Target I(q) — normalise to match our working scale
    I_target = I_data.copy()
    I_target = I_target / max(I_target.max(), 1e-12)

    # ── Initialise density from low-pass filtered random + radial hint ──
    rng = np.random.default_rng(SEED + 200)

    # Start with a Gaussian blob as initial guess (generic compact object)
    z, y, x = np.mgrid[:N, :N, :N] - N // 2
    r2 = x**2 + y**2 + z**2
    rho = np.exp(-r2 / (2 * (N / 6)**2)).astype(np.float64)
    # Add small random perturbation to break symmetry
    rho += 0.05 * rng.random((N, N, N))
    rho = np.maximum(rho, 0)
    rho = rho / max(rho.max(), 1e-12)

    # Support mask
    support = r2 < (N // 2 - 1)**2

    # ── Forward + gradient helpers ──
    def compute_Iq_and_grad(rho_in):
        """Compute I(q) from rho and gradient of L w.r.t. rho."""
        F = fftshift(fftn(ifftshift(rho_in)))
        I_3d = np.abs(F)**2

        # Radial average → I_model(q)
        I_model = np.zeros(N_Q)
        for i in range(N_Q):
            if bin_count[i] > 0:
                I_model[i] = np.mean(I_3d[bin_index == i])

        # Normalise model I(q) to same scale
        I_model_max = max(I_model.max(), 1e-12)
        I_model_norm = I_model / I_model_max

        # Residual per bin
        residual_per_bin = I_model_norm - I_target   # (N_Q,)

        # Loss = sum of squared residuals + Tikhonov
        alpha_smooth = 1e-4
        loss = np.sum(residual_per_bin**2) + alpha_smooth * np.sum(rho_in**2)

        # Gradient: ∂L/∂ρ via adjoint
        dL_dI3d = np.zeros_like(I_3d)
        for i in range(N_Q):
            if bin_count[i] > 0:
                mask_i = bin_index == i
                dL_dI3d[mask_i] = (2.0 / I_model_max) * residual_per_bin[i] / bin_count[i]

        # Chain rule through |F|^2 = F * conj(F)
        grad_F = dL_dI3d * np.conj(F)

        # Adjoint of FFT
        grad_rho = 2.0 * np.real(fftshift(ifftn(ifftshift(grad_F)))) * N**3
        # Tikhonov gradient
        grad_rho += 2 * alpha_smooth * rho_in

        return loss, I_model_norm, grad_rho

    # ── L-BFGS-B optimisation ──
    from scipy.optimize import minimize as sp_minimize

    support_flat = support.ravel()
    n_support = int(support_flat.sum())

    def run_lbfgsb(seed_offset):
        """Single L-BFGS-B run with given seed offset."""
        rng_run = np.random.default_rng(SEED + 200 + seed_offset)
        # Start with Gaussian blob + random perturbation
        rho_init = np.exp(-r2 / (2 * (N / 6)**2)).astype(np.float64)
        rho_init += 0.05 * rng_run.random((N, N, N))
        rho_init = np.maximum(rho_init, 0)
        rho_init = rho_init / max(rho_init.max(), 1e-12)

        x0 = rho_init.ravel()[support_flat].astype(np.float64)
        eval_count = [0]

        def obj(x_flat):
            rho_full = np.zeros(N**3, dtype=np.float64)
            rho_full[support_flat] = x_flat
            rho_3d = rho_full.reshape((N, N, N))
            loss, _, grad_3d = compute_Iq_and_grad(rho_3d)
            grad_flat = grad_3d.ravel()[support_flat].copy()
            eval_count[0] += 1
            return float(loss), np.ascontiguousarray(grad_flat, dtype=np.float64)

        result = sp_minimize(
            obj, x0, method='L-BFGS-B', jac=True,
            bounds=[(0, None)] * n_support,
            options={'maxiter': 1000, 'maxfun': 20000,
                     'ftol': 0, 'gtol': 1e-12, 'maxcor': 20}
        )
        return result

    print(f"[RECON] Running {N_RUNS} L-BFGS-B runs on {n_support} support voxels ...")

    best_result = None
    best_loss = float('inf')

    for run_idx in range(N_RUNS):
        result = run_lbfgsb(run_idx * 13)
        print(f"[RECON] Run {run_idx+1}/{N_RUNS}: loss={result.fun:.6f}  "
              f"iters={result.nit}  fevals={result.nfev}")
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    print(f"[RECON] Best loss: {best_loss:.6f}")

    rho_full = np.zeros(N**3)
    rho_full[support_flat] = best_result.x
    rho = rho_full.reshape((N, N, N))

    # Normalise
    if rho.max() > 0:
        rho = rho / rho.max()

    print(f"[RECON] Final range: [{rho.min():.4f}, {rho.max():.4f}]")
    return rho


# ═══════════════════════════════════════════════════════════
# 6. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(density_gt, density_rec, I_clean, q, density_rec_full=None):
    """Compute 3D density reconstruction metrics."""
    # Normalise
    gt = density_gt / max(density_gt.max(), 1e-12)
    rec = density_rec / max(density_rec.max(), 1e-12)

    # Ensure same shape
    s = min(gt.shape[0], rec.shape[0])
    gt = gt[:s, :s, :s]
    rec = rec[:s, :s, :s]

    # 3D CC
    cc_vol = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])
    re_vol = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    # Central slice metrics
    mid = s // 2
    gt_slice = gt[mid, :, :]
    rec_slice = rec[mid, :, :]
    dr = gt_slice.max() - gt_slice.min()
    if dr < 1e-12:
        dr = 1.0
    mse = np.mean((gt_slice - rec_slice)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_slice, rec_slice, data_range=dr))
    cc_slice = float(np.corrcoef(gt_slice.ravel(), rec_slice.ravel())[0, 1])

    # I(q) fit
    _, I_rec = forward_operator(rec * density_rec.max(), VOXEL_SIZE, N_Q, Q_MAX)
    cc_Iq = float(np.corrcoef(I_clean, I_rec)[0, 1])

    return {
        "PSNR_slice": psnr, "SSIM_slice": ssim_val,
        "CC_slice": cc_slice, "CC_volume": cc_vol,
        "RE_volume": re_vol, "CC_Iq": cc_Iq,
    }


# ═══════════════════════════════════════════════════════════
# 7. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(q, I_clean, I_noisy, density_gt, density_rec, metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mid = density_gt.shape[0] // 2
    gt_n = density_gt / max(density_gt.max(), 1e-12)
    rec_n = density_rec / max(density_rec.max(), 1e-12)
    s = min(gt_n.shape[0], rec_n.shape[0])

    for i, (title, data) in enumerate([
        ('GT (z-slice)', gt_n[min(mid, s-1)]),
        ('Recon (z-slice)', rec_n[min(mid, s-1)]),
        ('Error', gt_n[min(mid, s-1)] - rec_n[min(mid, s-1)]),
    ]):
        axes[0, i].imshow(data, cmap='hot' if i < 2 else 'RdBu_r', origin='lower')
        axes[0, i].set_title(title)

    axes[1, 0].semilogy(q, I_clean, 'b-', lw=2, label='GT')
    axes[1, 0].semilogy(q, I_noisy, 'k.', ms=3, alpha=0.5, label='Noisy')
    axes[1, 0].set_xlabel('q [Å⁻¹]'); axes[1, 0].set_ylabel('I(q)')
    axes[1, 0].set_title('SAXS Profile'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gt_n[min(mid, s-1), min(mid, s-1), :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(rec_n[min(mid, s-1), min(mid, s-1), :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title('1D Line Profile'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].text(0.5, 0.5, '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
                    transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=11,
                    family='monospace')
    axes[1, 2].set_title('Metrics'); axes[1, 2].axis('off')

    fig.suptitle(f"DENSS — SAXS Electron Density Reconstruction\n"
                 f"PSNR={metrics['PSNR_slice']:.1f} dB  |  SSIM={metrics['SSIM_slice']:.4f}  |  "
                 f"CC={metrics['CC_volume']:.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[VIS] Saved → {save_path}")


if __name__ == "__main__":
    print("=" * 65)
    print("  DENSS — SAXS Density Reconstruction")
    print("=" * 65)
    q, I_noisy, I_clean, density_gt = load_or_generate_data()
    density_rec = reconstruct(q, I_noisy, density_gt.shape)
    metrics = compute_metrics(density_gt, density_rec, I_clean, q)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), density_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), density_gt)
    visualize_results(q, I_clean, I_noisy, density_gt, density_rec, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)
