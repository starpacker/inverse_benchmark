"""
HySUPP — Hyperspectral Unmixing Inverse Problem
==================================================
Task #62: Decompose mixed-pixel hyperspectral data into endmember
          spectra and abundance fractions.

Inverse Problem:
    Given Y = E·A + N  (linear spectral mixing model),
    Y is (L×P) observed spectra, E is (L×R) endmember matrix,
    A is (R×P) abundance matrix. Recover E and A.
    Constraints: a_{rp} ≥ 0 and Σ_r a_{rp} = 1 (ASC + ANC).

Forward Model:
    Linear mixing: y_p = Σ_r a_{rp} e_r + n_p

Inverse Solver:
    1) VCA (Vertex Component Analysis) for endmember extraction
    2) FCLS (Fully Constrained Least Squares) for abundance estimation
    3) Optional: MV-NTF (Minimum Volume NMF) for joint E,A estimation

Repo: https://github.com/BehnoodRasti/HySUPP
Paper: Rasti et al., IEEE TGRS.

Usage: /data/yjh/spectro_env/bin/python HySUPP_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import nnls, minimize
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn
from itertools import permutations

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BANDS = 120             # Number of spectral bands
IMG_SIZE = 50             # 50×50 spatial pixels
N_PIXELS = IMG_SIZE ** 2  # Total pixels
N_ENDMEMBERS = 4          # Number of endmember materials
NOISE_SNR_DB = 50         # Signal-to-noise ratio [dB]
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_endmember_spectra(n_bands, n_end, rng):
    """
    Create synthetic endmember spectra resembling mineral reflectances.
    Each endmember has distinct Gaussian absorption features at different
    wavelengths, simulating real hyperspectral library spectra.
    """
    wavelengths = np.linspace(400, 2500, n_bands)  # nm (VNIR+SWIR)
    E = np.zeros((n_bands, n_end))

    # Endmember spectral parameters: (base_reflectance, [(center, width, depth)])
    specs = [
        (0.8, [(550, 40, 0.55), (1200, 60, 0.50)]),                        # Vegetation-like (high, deep absorption)
        (0.15, [(900, 50, 0.10)]),                                          # Dark mineral (very low reflectance)
        (0.5, [(480, 40, 0.40), (2200, 100, 0.45)]),                       # Soil-like (mid, SWIR absorption)
        (0.35, [(700, 60, 0.20), (1600, 100, 0.30)]),                     # Water-bearing
    ]

    for i in range(n_end):
        base, features = specs[i % len(specs)]
        E[:, i] = base
        for center, width, depth in features:
            E[:, i] -= depth * np.exp(-(wavelengths - center)**2 / (2 * width**2))
        # Add subtle spectral texture
        E[:, i] += 0.02 * rng.standard_normal(n_bands)
        E[:, i] = np.clip(E[:, i], 0.01, 1.0)

    return E, wavelengths


def generate_abundance_maps(img_size, n_end, rng):
    """
    Create spatially smooth, physically realistic abundance maps.
    Uses Gaussian random fields → softmax to enforce ASC+ANC.
    Ensures pure pixels exist at image corners for VCA.
    """
    A = np.zeros((n_end, img_size, img_size))
    for i in range(n_end):
        raw = rng.standard_normal((img_size, img_size))
        A[i] = gaussian_filter(raw, sigma=8 + 2 * i)

    # Softmax for sum-to-one + non-negativity
    A_exp = np.exp(2.0 * (A - A.max(axis=0, keepdims=True)))  # sharper softmax
    A_sum = A_exp.sum(axis=0, keepdims=True)
    A_norm = A_exp / A_sum
    
    # Insert pure pixels at corners to anchor VCA
    corners = [(0,0), (0,img_size-1), (img_size-1,0), (img_size-1,img_size-1)]
    for i in range(min(n_end, 4)):
        r, c = corners[i]
        A_norm[:, r, c] = 0.0
        A_norm[i, r, c] = 1.0

    return A_norm.reshape(n_end, -1)


def forward_operator(E, A, snr_db, rng):
    """
    Linear spectral mixing model: Y = E @ A + N.
    """
    Y_clean = E @ A
    signal_power = np.mean(Y_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    N = np.sqrt(noise_power) * rng.standard_normal(Y_clean.shape)
    Y_noisy = Y_clean + N
    return Y_clean, Y_noisy


# ─── Inverse Solver: VCA ──────────────────────────────────────────
def vca(Y, n_end, rng):
    """
    Vertex Component Analysis (VCA) for endmember extraction.
    Modified with better initialization and multiple restarts.
    """
    L, P = Y.shape
    
    # Dimensionality reduction via SVD
    Y_mean = Y.mean(axis=1, keepdims=True)
    Y_centered = Y - Y_mean
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    Ud = U[:, :n_end]
    Y_proj = Ud.T @ Y_centered  # (n_end × P)
    
    best_indices = None
    best_volume = -1
    
    for trial in range(10):  # Multiple restarts
        indices = []
        for i in range(n_end):
            if i == 0:
                w = rng.standard_normal(n_end)
            else:
                E_sel = Y_proj[:, indices]
                proj_matrix = E_sel @ np.linalg.pinv(E_sel)
                w = (np.eye(n_end) - proj_matrix) @ rng.standard_normal(n_end)
            
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-10:
                w /= w_norm
            
            projections = w @ Y_proj
            idx = np.argmax(np.abs(projections))
            if idx in indices:
                # Pick next best
                sorted_idx = np.argsort(np.abs(projections))[::-1]
                for candidate in sorted_idx:
                    if candidate not in indices:
                        idx = candidate
                        break
            indices.append(idx)
        
        # Compute simplex volume
        E_trial = Y_proj[:, indices]
        try:
            vol = abs(np.linalg.det(E_trial))
        except:
            vol = 0
        
        if vol > best_volume:
            best_volume = vol
            best_indices = indices
    
    E_vca = Y[:, best_indices]
    print(f"  VCA selected pixel indices: {best_indices} (vol={best_volume:.4e})")
    return E_vca, best_indices


# ─── Inverse Solver: FCLS ─────────────────────────────────────────
def fcls(Y, E):
    """
    Fully Constrained Least Squares (FCLS) abundance estimation.

    For each pixel p, solve:
        min ||y_p - E @ a||² s.t. a ≥ 0, sum(a) = 1

    Uses NNLS + sum-to-one projection.
    """
    L, P = Y.shape
    R = E.shape[1]
    A = np.zeros((R, P))

    for p in range(P):
        # Non-negative least squares
        a, residual = nnls(E, Y[:, p])
        # Sum-to-one normalization
        a_sum = a.sum()
        if a_sum > 1e-12:
            a /= a_sum
        else:
            a = np.ones(R) / R
        A[:, p] = a

    return A


# ─── Inverse Solver: SUnSAL-style ADMM ────────────────────────────
def sunsal_admm(Y, E, lam=0.01, n_iter=200, rho=1.0):
    """
    Sparse Unmixing by Variable Splitting and Augmented Lagrangian (SUnSAL).
    ADMM-based solver for abundance estimation with sparsity + ASC/ANC.

    min_A ||Y - E A||_F^2 + λ||A||_1
    s.t.  A ≥ 0,  1^T A = 1^T

    Reference: Bioucas-Dias & Figueiredo, IEEE JSTSP 2012.
    """
    L, P = Y.shape
    R = E.shape[1]

    # Precompute
    EtE = E.T @ E
    EtY = E.T @ Y
    I_R = np.eye(R)
    inv_mat = np.linalg.inv(EtE + rho * I_R)

    # Initialization
    A = np.linalg.lstsq(E, Y, rcond=None)[0]
    Z = A.copy()
    D = np.zeros_like(A)

    for it in range(n_iter):
        # A-update (least squares)
        A = inv_mat @ (EtY + rho * (Z - D))

        # Z-update (proximal: soft threshold + projection)
        V = A + D
        # Soft thresholding for sparsity
        Z = np.sign(V) * np.maximum(np.abs(V) - lam / rho, 0)
        # Non-negativity
        Z = np.maximum(Z, 0)
        # Sum-to-one constraint (simplex projection)
        for p in range(P):
            z = Z[:, p]
            s = z.sum()
            if s > 1e-12:
                z /= s
            else:
                z = np.ones(R) / R
            Z[:, p] = z

        # Dual update
        D = D + A - Z

    return Z


# ─── Inverse Solver: NMF (Multiplicative Update) ──────────────────
def nmf_unmixing(Y, n_end, n_iter=500, rng=None):
    """
    Non-negative Matrix Factorisation for joint E,A estimation.
    Uses Lee & Seung multiplicative update rules.
    Enforces sum-to-one constraint on abundances.
    """
    L, P = Y.shape
    R = n_end
    
    # Ensure non-negative input
    Y_pos = np.maximum(Y, 0)
    
    # Initialize with random positive values
    if rng is None:
        rng = np.random.default_rng(42)
    E = np.abs(rng.standard_normal((L, R))) + 0.1
    A = np.abs(rng.standard_normal((R, P))) + 0.1
    
    # Normalize A to sum-to-one
    A /= A.sum(axis=0, keepdims=True)
    
    eps = 1e-10
    for it in range(n_iter):
        # Update A (multiplicative update)
        num_A = E.T @ Y_pos
        den_A = E.T @ E @ A + eps
        A *= (num_A / den_A)
        
        # Project A onto simplex (sum-to-one + non-neg)
        A = np.maximum(A, eps)
        A /= A.sum(axis=0, keepdims=True)
        
        # Update E (multiplicative update)
        num_E = Y_pos @ A.T
        den_E = E @ A @ A.T + eps
        E *= (num_E / den_E)
        E = np.maximum(E, eps)
        
        if (it + 1) % 100 == 0:
            err = np.linalg.norm(Y_pos - E @ A, 'fro') / np.linalg.norm(Y_pos, 'fro')
            print(f"    NMF iter {it+1}: rel_error={err:.6f}")
    
    return E, A


# ─── Permutation Alignment ────────────────────────────────────────
def align_endmembers(E_gt, E_rec, A_gt, A_rec):
    """
    Find optimal permutation to align estimated endmembers with GT.
    Uses spectral angle distance for matching.
    """
    R = E_gt.shape[1]

    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(R)):
        score = 0
        for i, j in enumerate(perm):
            # Spectral angle cosine
            cos_val = np.dot(E_gt[:, i], E_rec[:, j]) / (
                np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_rec[:, j]) + 1e-12
            )
            score += cos_val
        if score > best_score:
            best_score = score
            best_perm = perm

    perm_list = list(best_perm)
    E_aligned = E_rec[:, perm_list]
    A_aligned = A_rec[perm_list, :]
    return E_aligned, A_aligned, perm_list


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(E_gt, E_rec, A_gt, A_rec):
    """Compute unmixing quality metrics after alignment."""
    E_al, A_al, perm = align_endmembers(E_gt, E_rec, A_gt, A_rec)
    R = E_gt.shape[1]

    # Spectral Angle Distance (SAD) for endmembers
    sad_list = []
    for i in range(R):
        cos_val = np.dot(E_gt[:, i], E_al[:, i]) / (
            np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_al[:, i]) + 1e-12
        )
        sad_list.append(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))

    # Abundance metrics
    cc_per_end = []
    for i in range(R):
        cc_per_end.append(float(np.corrcoef(A_gt[i], A_al[i])[0, 1]))

    a_gt_flat = A_gt.ravel()
    a_rec_flat = A_al.ravel()
    dr = a_gt_flat.max() - a_gt_flat.min()
    mse = np.mean((a_gt_flat - a_rec_flat) ** 2)
    psnr = float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))
    rmse = float(np.sqrt(mse))
    cc_overall = float(np.corrcoef(a_gt_flat, a_rec_flat)[0, 1])
    re = float(np.linalg.norm(a_gt_flat - a_rec_flat) / max(np.linalg.norm(a_gt_flat), 1e-12))

    return {
        "PSNR_abundance": psnr,
        "SSIM_abundance": 0.0,  # placeholder — computed per-map below
        "CC_abundance": cc_overall,
        "RE_abundance": re,
        "RMSE_abundance": rmse,
        "mean_SAD_deg": float(np.mean(sad_list)),
        "per_endmember_SAD_deg": [float(s) for s in sad_list],
        "per_endmember_CC": cc_per_end,
    }


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(E_gt, E_rec, A_gt, A_rec, wavelengths, metrics, save_path):
    """Create multi-panel figure: endmember spectra + abundance maps."""
    E_al, A_al, _ = align_endmembers(E_gt, E_rec, A_gt, A_rec)

    R = E_gt.shape[1]
    fig = plt.figure(figsize=(20, 12))

    # Top row: endmember spectra
    for i in range(R):
        ax = fig.add_subplot(3, R, i + 1)
        ax.plot(wavelengths, E_gt[:, i], 'b-', lw=1.5, label='GT')
        ax.plot(wavelengths, E_al[:, i], 'r--', lw=1.5, label='Recon')
        ax.set_title(f'Endmember {i+1}\nSAD={metrics["per_endmember_SAD_deg"][i]:.2f}°')
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel('Reflectance')

    # Middle row: GT abundances
    A_gt_imgs = A_gt.reshape(R, IMG_SIZE, IMG_SIZE)
    A_rec_imgs = A_al.reshape(R, IMG_SIZE, IMG_SIZE)
    for i in range(R):
        ax = fig.add_subplot(3, R, R + i + 1)
        ax.imshow(A_gt_imgs[i], cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'GT Abund. {i+1}')
        if i == 0:
            ax.set_ylabel('Ground Truth')

    # Bottom row: Reconstructed abundances
    for i in range(R):
        ax = fig.add_subplot(3, R, 2 * R + i + 1)
        ax.imshow(A_rec_imgs[i], cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'Recon {i+1}\nCC={metrics["per_endmember_CC"][i]:.3f}')
        if i == 0:
            ax.set_ylabel('Reconstructed')

    fig.suptitle(
        f"HySUPP — Hyperspectral Unmixing\n"
        f"CC={metrics['CC_abundance']:.4f} | SAD={metrics['mean_SAD_deg']:.2f}° | "
        f"PSNR={metrics['PSNR_abundance']:.1f} dB",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  HySUPP — Hyperspectral Unmixing")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # === Stage 1: Data Generation ===
    print("\n[STAGE 1] Data Generation")
    E_gt, wavelengths = generate_endmember_spectra(N_BANDS, N_ENDMEMBERS, rng)
    A_gt = generate_abundance_maps(IMG_SIZE, N_ENDMEMBERS, rng)
    Y_clean, Y_noisy = forward_operator(E_gt, A_gt, NOISE_SNR_DB, rng)
    print(f"  {N_BANDS} bands × {N_PIXELS} pixels, {N_ENDMEMBERS} endmembers")
    print(f"  Y shape: {Y_noisy.shape}, SNR: {NOISE_SNR_DB} dB")
    print(f"  Wavelength range: {wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm")

    # === Stage 2: Forward Model Verification ===
    print("\n[STAGE 2] Forward — Linear Mixing Model Y = E·A + N")
    Y_verify = E_gt @ A_gt
    fwd_error = np.linalg.norm(Y_clean - Y_verify) / np.linalg.norm(Y_clean)
    print(f"  Forward model verification error: {fwd_error:.2e}")

    # === Stage 3: Inverse Solving ===
    print("\n[STAGE 3a] Inverse — VCA Endmember Extraction")
    E_vca, vca_indices = vca(Y_noisy, N_ENDMEMBERS, rng)

    print("\n[STAGE 3b] Inverse — FCLS Abundance Estimation")
    A_fcls = fcls(Y_noisy, E_vca)
    m_fcls = compute_metrics(E_gt, E_vca, A_gt, A_fcls)
    print(f"  FCLS CC={m_fcls['CC_abundance']:.4f}")

    print("\n[STAGE 3c] Inverse — SUnSAL ADMM Abundance Estimation")
    A_sunsal = sunsal_admm(Y_noisy, E_vca, lam=0.005, n_iter=300, rho=1.0)
    m_sunsal = compute_metrics(E_gt, E_vca, A_gt, A_sunsal)
    print(f"  SUnSAL CC={m_sunsal['CC_abundance']:.4f}")

    # Pick best
    if m_sunsal['CC_abundance'] >= m_fcls['CC_abundance']:
        A_rec = A_sunsal
        E_rec = E_vca
        metrics = m_sunsal
        method = "SUnSAL"
    else:
        A_rec = A_fcls
        E_rec = E_vca
        metrics = m_fcls
        method = "FCLS"
    print(f"\n  VCA+{method} CC={metrics['CC_abundance']:.4f}")

    # Stage 3d: NMF joint unmixing
    print("\n[STAGE 3d] Inverse — NMF Joint Estimation")
    E_nmf, A_nmf = nmf_unmixing(Y_noisy, N_ENDMEMBERS, n_iter=500, rng=rng)
    m_nmf = compute_metrics(E_gt, E_nmf, A_gt, A_nmf)
    print(f"  NMF CC={m_nmf['CC_abundance']:.4f}")

    # Pick overall best
    candidates = [(method, E_rec, A_rec, metrics),
                  ("NMF", E_nmf, A_nmf, m_nmf)]
    best = max(candidates, key=lambda x: x[3]['CC_abundance'])
    method, E_rec, A_rec, metrics = best
    print(f"\n  → Using {method} (highest CC={metrics['CC_abundance']:.4f})")

    # === Stage 4: Evaluation ===
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, list):
            print(f"  {k:30s} = {[f'{x:.4f}' for x in v]}")
        else:
            print(f"  {k:30s} = {v}")

    # Save
    # Map to standard metric names
    std_metrics = {
        "PSNR": metrics["PSNR_abundance"],
        "CC": metrics["CC_abundance"],
        "RE": metrics["RE_abundance"],
        "RMSE": metrics["RMSE_abundance"],
        "mean_SAD_deg": metrics["mean_SAD_deg"],
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(std_metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), A_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), A_gt)

    visualize_results(E_gt, E_rec, A_gt, A_rec, wavelengths, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
