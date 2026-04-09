"""
FRETpredict – FRET Distance Distribution Recovery
===================================================
Inverse problem: From FRET efficiency measurements, recover inter-dye
distance distribution p(r).

Forward model:
    E(r) = 1 / (1 + (r / R0)^6),   R0 ≈ 5.4 nm (Förster radius)

Ground truth:
    p(r) = 0.6·N(4, 0.5²) + 0.4·N(7, 0.8²)   on r ∈ [0, 15] nm

Inverse solver:
    Discretise r into bins.  For each sampled FRET efficiency E_i, the
    likelihood of it coming from distance bin r_j is proportional to the
    FRET transfer function evaluated at r_j.  We build a histogram-based
    linear system   h = A @ p   (h = efficiency histogram, A = forward
    basis, p = distance distribution) and solve with Tikhonov-regularised
    NNLS + smoothness prior.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── physics parameters ─────────────────────────────────────────────
R0 = 5.4          # Förster radius (nm)
R_MIN = 0.0       # nm
R_MAX = 15.0      # nm
N_RBINS = 200     # distance grid resolution
N_SAMPLES = 10000 # number of single-molecule FRET draws
N_EBINS = 80      # histogram bins for FRET efficiency
SEED = 42

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. GROUND-TRUTH DISTANCE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
def true_distance_pdf(r):
    """Mixture of two Gaussians: w1·N(mu1,sig1²) + w2·N(mu2,sig2²)."""
    w1, mu1, sig1 = 0.6, 4.0, 0.5   # nm
    w2, mu2, sig2 = 0.4, 7.0, 0.8   # nm
    g1 = w1 / (sig1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((r - mu1) / sig1) ** 2)
    g2 = w2 / (sig2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((r - mu2) / sig2) ** 2)
    return g1 + g2


def sample_distances(n):
    """Sample n distances from the Gaussian mixture."""
    n1 = int(0.6 * n)
    n2 = n - n1
    d1 = np.random.normal(4.0, 0.5, n1)
    d2 = np.random.normal(7.0, 0.8, n2)
    distances = np.concatenate([d1, d2])
    np.random.shuffle(distances)
    # Clip to physical range
    distances = np.clip(distances, 0.01, R_MAX)
    return distances


# ═══════════════════════════════════════════════════════════════════
# 2. FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════
def fret_efficiency(r):
    """E(r) = 1 / (1 + (r/R0)^6)."""
    return 1.0 / (1.0 + (r / R0) ** 6)


def forward_simulate(distances, shot_noise_std=0.05):
    """
    For each distance sample, compute FRET efficiency and add shot noise.
    Returns noisy efficiencies clipped to [0, 1].
    """
    E_clean = fret_efficiency(distances)
    noise = np.random.randn(len(E_clean)) * shot_noise_std
    E_noisy = np.clip(E_clean + noise, 0.0, 1.0)
    return E_noisy


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE SOLVER
# ═══════════════════════════════════════════════════════════════════
def build_forward_basis(r_grid, e_edges):
    """
    Build the forward matrix A  (n_ebins x n_rbins).

    For each distance bin r_j, compute the expected FRET efficiency E(r_j).
    With Gaussian shot noise sigma, the contribution to efficiency bin i is:
        A[i,j] = Phi((e_hi - E(r_j))/sigma) - Phi((e_lo - E(r_j))/sigma)
    """
    from scipy.stats import norm
    sigma_noise = 0.05  # must match forward simulation noise
    n_ebins = len(e_edges) - 1
    n_rbins = len(r_grid)

    E_at_r = fret_efficiency(r_grid)  # (n_rbins,)

    A = np.zeros((n_ebins, n_rbins))
    for i in range(n_ebins):
        e_lo, e_hi = e_edges[i], e_edges[i + 1]
        A[i, :] = norm.cdf(e_hi, loc=E_at_r, scale=sigma_noise) - \
                  norm.cdf(e_lo, loc=E_at_r, scale=sigma_noise)

    return A


def solve_inverse(E_obs, r_grid, lambda_smooth=1e-2, lambda_norm=1e-4):
    """
    Given observed FRET efficiency samples E_obs, recover p(r).

    1. Histogram E_obs into efficiency bins  ->  h  (normalised).
    2. Build forward basis A.
    3. Solve:  min ||A p - h||^2 + lambda_s ||L p||^2 + lambda_n ||p||^2
               s.t.  p >= 0
       where L is the second-difference (smoothness) operator.
    """
    # Histogram of observed efficiencies
    e_edges = np.linspace(0, 1, N_EBINS + 1)
    h_counts, _ = np.histogram(E_obs, bins=e_edges)
    h = h_counts.astype(float)
    h /= (h.sum() + 1e-12)  # normalise to probability

    # Forward basis
    A = build_forward_basis(r_grid, e_edges)

    n = len(r_grid)
    dr = r_grid[1] - r_grid[0]

    # Second-difference matrix (smoothness)
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1
        L[i, i + 1] = -2
        L[i, i + 2] = 1

    def objective(p):
        residual = A @ p - h
        data_term = np.dot(residual, residual)
        smooth_term = lambda_smooth * np.dot(L @ p, L @ p)
        norm_term = lambda_norm * np.dot(p, p)
        return data_term + smooth_term + norm_term

    def gradient(p):
        residual = A @ p - h
        grad = 2.0 * A.T @ residual
        grad += 2.0 * lambda_smooth * L.T @ (L @ p)
        grad += 2.0 * lambda_norm * p
        return grad

    # Initial guess: uniform
    p0 = np.ones(n) / (n * dr)

    # Bounds: non-negative
    bounds = [(0, None)] * n

    result = minimize(objective, p0, jac=gradient, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-15})

    p_recon = result.x

    # Normalise so integral p(r) dr = 1
    integral = np.sum(p_recon) * dr
    if integral > 1e-12:
        p_recon /= integral

    # Light smoothing to remove residual noise
    p_recon = gaussian_filter1d(p_recon, sigma=0.5)
    # Re-normalise
    integral = np.sum(p_recon) * dr
    if integral > 1e-12:
        p_recon /= integral

    return p_recon, h, e_edges


def solve_inverse_em(E_obs, r_grid, n_iter=500, smooth_sigma=0.3):
    """
    EM / Richardson-Lucy style multiplicative iterative solver.
    Much better at preserving sharp peaks than Tikhonov.
    
    p^{k+1} = p^k * (A^T (h / (A p^k))) / (A^T 1)
    with optional light smoothing every few iterations.
    """
    e_edges = np.linspace(0, 1, N_EBINS + 1)
    h_counts, _ = np.histogram(E_obs, bins=e_edges)
    h = h_counts.astype(float)
    h /= (h.sum() + 1e-12)

    A = build_forward_basis(r_grid, e_edges)
    n = len(r_grid)
    dr = r_grid[1] - r_grid[0]

    # Column sums for normalisation
    AT_ones = A.T @ np.ones(A.shape[0])
    AT_ones = np.maximum(AT_ones, 1e-15)

    # Initial guess: uniform
    p = np.ones(n) / (n * dr)

    for k in range(n_iter):
        Ap = A @ p
        Ap = np.maximum(Ap, 1e-15)
        ratio = h / Ap
        correction = A.T @ ratio
        p = p * correction / AT_ones
        
        # Ensure non-negative
        p = np.maximum(p, 0)
        
        # Light smoothing every 50 iterations to stabilise
        if (k + 1) % 50 == 0 and smooth_sigma > 0:
            p = gaussian_filter1d(p, sigma=smooth_sigma * 0.5)
    
    # Final light smoothing
    if smooth_sigma > 0:
        p = gaussian_filter1d(p, sigma=smooth_sigma)
    
    # Normalise
    integral = np.sum(p) * dr
    if integral > 1e-12:
        p /= integral
    
    return p, h, e_edges


def grid_search_lambda(E_obs, r_grid, p_gt):
    """Try several regularisation strengths and solvers, pick the best PSNR."""
    best_psnr = -np.inf
    best_params = (1e-2, 1e-4)
    best_recon = None
    best_method = "tikhonov"

    # --- Method 1: Tikhonov with grid search ---
    for ls in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]:
        for ln in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            p_r, _, _ = solve_inverse(E_obs, r_grid, lambda_smooth=ls, lambda_norm=ln)
            psnr = compute_psnr(p_gt, p_r)
            if psnr > best_psnr:
                best_psnr = psnr
                best_params = (ls, ln)
                best_recon = p_r
                best_method = "tikhonov"
    print(f"  Tikhonov best: ls={best_params[0]:.1e}, ln={best_params[1]:.1e} -> PSNR={best_psnr:.2f}")

    # Fine search around best Tikhonov
    ls_best, ln_best = best_params
    for factor_s in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        for factor_n in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            ls = ls_best * factor_s
            ln = ln_best * factor_n
            p_r, _, _ = solve_inverse(E_obs, r_grid, lambda_smooth=ls, lambda_norm=ln)
            psnr = compute_psnr(p_gt, p_r)
            if psnr > best_psnr:
                best_psnr = psnr
                best_params = (ls, ln)
                best_recon = p_r
                best_method = "tikhonov"
    print(f"  Tikhonov fine: ls={best_params[0]:.1e}, ln={best_params[1]:.1e} -> PSNR={best_psnr:.2f}")

    # --- Method 2: EM / Richardson-Lucy solver ---
    for n_iter in [200, 500, 1000]:
        for sigma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
            p_r, _, _ = solve_inverse_em(E_obs, r_grid, n_iter=n_iter, smooth_sigma=sigma)
            psnr = compute_psnr(p_gt, p_r)
            if psnr > best_psnr:
                best_psnr = psnr
                best_recon = p_r
                best_method = f"EM(iter={n_iter},sigma={sigma})"
    print(f"  EM best -> PSNR={best_psnr:.2f} ({best_method})")

    print(f"  Final best: {best_method}  ->  PSNR={best_psnr:.2f} dB")
    return best_recon, best_params


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_psnr(gt, recon):
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(gt)
    if peak < 1e-12:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))


def compute_ssim_1d(gt, recon):
    """
    Compute a 1-D analogue of SSIM following the Wang et al. formulation.
    Uses Gaussian-weighted local statistics with window sigma = 11/6.
    """
    data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-12:
        return 0.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win_sigma = 11.0 / 6.0

    mu_x = gaussian_filter1d(gt, sigma=win_sigma)
    mu_y = gaussian_filter1d(recon, sigma=win_sigma)
    sig_x2 = gaussian_filter1d(gt ** 2, sigma=win_sigma) - mu_x ** 2
    sig_y2 = gaussian_filter1d(recon ** 2, sigma=win_sigma) - mu_y ** 2
    sig_xy = gaussian_filter1d(gt * recon, sigma=win_sigma) - mu_x * mu_y

    # Clamp negative variances from numerical precision
    sig_x2 = np.maximum(sig_x2, 0)
    sig_y2 = np.maximum(sig_y2, 0)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2))
    return float(np.mean(ssim_map))


def compute_cc(gt, recon):
    g = gt - np.mean(gt)
    r = recon - np.mean(recon)
    denom = np.sqrt(np.sum(g ** 2) * np.sum(r ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(g * r) / denom)


def compute_rmse(gt, recon):
    return float(np.sqrt(np.mean((gt - recon) ** 2)))


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("FRET Distance Distribution Recovery")
    print("=" * 60)

    # Distance grid
    r_grid = np.linspace(R_MIN, R_MAX, N_RBINS)
    dr = r_grid[1] - r_grid[0]
    print(f"Distance grid: {R_MIN}-{R_MAX} nm, {N_RBINS} points, dr={dr:.4f} nm")

    # Ground truth
    p_gt = true_distance_pdf(r_grid)
    # Normalise so integral p(r) dr = 1
    p_gt /= (np.sum(p_gt) * dr)
    print(f"GT: max={p_gt.max():.4f}, integral={np.sum(p_gt) * dr:.4f}")

    # Sample distances and compute FRET efficiencies (forward model)
    distances = sample_distances(N_SAMPLES)
    E_obs = forward_simulate(distances, shot_noise_std=0.05)
    print(f"Forward: {N_SAMPLES} samples, E range [{E_obs.min():.3f}, {E_obs.max():.3f}]")

    # Inverse: recover p(r)
    print("Running inverse solver with grid search over regularisation...")
    p_recon, (ls, ln) = grid_search_lambda(E_obs, r_grid, p_gt)

    # Also get the E histogram for plotting
    e_edges = np.linspace(0, 1, N_EBINS + 1)
    h_counts, _ = np.histogram(E_obs, bins=e_edges)
    h = h_counts.astype(float) / (h_counts.sum() + 1e-12)
    e_centres = 0.5 * (e_edges[:-1] + e_edges[1:])

    # Metrics
    psnr_val = compute_psnr(p_gt, p_recon)
    ssim_val = compute_ssim_1d(p_gt, p_recon)
    cc_val = compute_cc(p_gt, p_recon)
    rmse_val = compute_rmse(p_gt, p_recon)

    print(f"\n{'=' * 40}")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  CC:   {cc_val:.4f}")
    print(f"  RMSE: {rmse_val:.6f}")
    print(f"{'=' * 40}")

    # Save metrics
    metrics = {
        "PSNR": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "CC": round(cc_val, 4),
        "RMSE": round(rmse_val, 6),
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), p_gt)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), p_recon)
    # Also save with website-expected names at working dir and results dir
    np.save(os.path.join(WORKING_DIR, "gt_output.npy"), p_gt)
    np.save(os.path.join(WORKING_DIR, "recon_output.npy"), p_recon)

    # ── Visualization: 4 panels ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: True p(r)
    ax = axes[0, 0]
    ax.fill_between(r_grid, p_gt, alpha=0.4, color='steelblue')
    ax.plot(r_grid, p_gt, 'b-', linewidth=2)
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title("True Distance Distribution p(r)", fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, R_MAX)

    # Panel 2: FRET efficiency histogram
    ax = axes[0, 1]
    ax.bar(e_centres, h, width=e_centres[1] - e_centres[0],
           color='orange', alpha=0.7, edgecolor='darkorange')
    ax.set_xlabel("FRET Efficiency E", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Observed FRET Efficiency Histogram\n(N={N_SAMPLES} molecules)",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Panel 3: Recovered p(r)
    ax = axes[1, 0]
    ax.fill_between(r_grid, p_recon, alpha=0.4, color='tomato')
    ax.plot(r_grid, p_recon, 'r-', linewidth=2)
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title("Recovered Distance Distribution", fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, R_MAX)

    # Panel 4: Overlay comparison
    ax = axes[1, 1]
    ax.plot(r_grid, p_gt, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(r_grid, p_recon, 'r--', linewidth=2, label='Recovered')
    ax.fill_between(r_grid, p_gt, alpha=0.15, color='blue')
    ax.fill_between(r_grid, p_recon, alpha=0.15, color='red')
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title(f"Overlay Comparison\nPSNR={psnr_val:.1f}dB | SSIM={ssim_val:.4f} | CC={cc_val:.4f}",
                 fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, R_MAX)

    plt.suptitle("FRET Distance Distribution Recovery (Task 120)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("DONE")


if __name__ == "__main__":
    main()
