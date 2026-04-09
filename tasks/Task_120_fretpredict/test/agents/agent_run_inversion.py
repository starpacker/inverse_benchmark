import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.optimize import minimize

from scipy.ndimage import gaussian_filter1d

from scipy.stats import norm

def fret_efficiency(r, R0):
    """E(r) = 1 / (1 + (r/R0)^6)."""
    return 1.0 / (1.0 + (r / R0) ** 6)

def build_forward_basis(r_grid, e_edges, R0, sigma_noise=0.05):
    """
    Build the forward matrix A (n_ebins x n_rbins).
    For each distance bin r_j, compute the expected FRET efficiency E(r_j).
    """
    n_ebins = len(e_edges) - 1
    n_rbins = len(r_grid)
    E_at_r = fret_efficiency(r_grid, R0)
    A = np.zeros((n_ebins, n_rbins))
    for i in range(n_ebins):
        e_lo, e_hi = e_edges[i], e_edges[i + 1]
        A[i, :] = norm.cdf(e_hi, loc=E_at_r, scale=sigma_noise) - \
                  norm.cdf(e_lo, loc=E_at_r, scale=sigma_noise)
    return A

def solve_inverse_tikhonov(E_obs, r_grid, n_ebins, R0, lambda_smooth=1e-2, lambda_norm=1e-4):
    """
    Tikhonov regularized solver for distance distribution recovery.
    """
    e_edges = np.linspace(0, 1, n_ebins + 1)
    h_counts, _ = np.histogram(E_obs, bins=e_edges)
    h = h_counts.astype(float)
    h /= (h.sum() + 1e-12)

    A = build_forward_basis(r_grid, e_edges, R0)
    n = len(r_grid)
    dr = r_grid[1] - r_grid[0]

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

    p0 = np.ones(n) / (n * dr)
    bounds = [(0, None)] * n

    result = minimize(objective, p0, jac=gradient, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-15})

    p_recon = result.x
    integral = np.sum(p_recon) * dr
    if integral > 1e-12:
        p_recon /= integral

    p_recon = gaussian_filter1d(p_recon, sigma=0.5)
    integral = np.sum(p_recon) * dr
    if integral > 1e-12:
        p_recon /= integral

    return p_recon, h, e_edges

def solve_inverse_em(E_obs, r_grid, n_ebins, R0, n_iter=500, smooth_sigma=0.3):
    """
    EM / Richardson-Lucy style multiplicative iterative solver.
    """
    e_edges = np.linspace(0, 1, n_ebins + 1)
    h_counts, _ = np.histogram(E_obs, bins=e_edges)
    h = h_counts.astype(float)
    h /= (h.sum() + 1e-12)

    A = build_forward_basis(r_grid, e_edges, R0)
    n = len(r_grid)
    dr = r_grid[1] - r_grid[0]

    AT_ones = A.T @ np.ones(A.shape[0])
    AT_ones = np.maximum(AT_ones, 1e-15)

    p = np.ones(n) / (n * dr)

    for k in range(n_iter):
        Ap = A @ p
        Ap = np.maximum(Ap, 1e-15)
        ratio = h / Ap
        correction = A.T @ ratio
        p = p * correction / AT_ones
        p = np.maximum(p, 0)
        if (k + 1) % 50 == 0 and smooth_sigma > 0:
            p = gaussian_filter1d(p, sigma=smooth_sigma * 0.5)

    if smooth_sigma > 0:
        p = gaussian_filter1d(p, sigma=smooth_sigma)

    integral = np.sum(p) * dr
    if integral > 1e-12:
        p /= integral

    return p, h, e_edges

def compute_psnr(gt, recon):
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(gt)
    if peak < 1e-12:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))

def run_inversion(E_obs, r_grid, p_gt, R0, n_ebins=80):
    """
    Run inverse solver with grid search over regularization parameters.
    
    Args:
        E_obs: observed FRET efficiencies
        r_grid: distance grid
        p_gt: ground truth for parameter selection
        R0: Förster radius
        n_ebins: number of efficiency histogram bins
        
    Returns:
        p_recon: recovered distance distribution
        best_params: tuple of best regularization parameters
        h: efficiency histogram
        e_edges: efficiency bin edges
    """
    best_psnr = -np.inf
    best_params = (1e-2, 1e-4)
    best_recon = None
    best_method = "tikhonov"

    # Tikhonov grid search
    for ls in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]:
        for ln in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            p_r, _, _ = solve_inverse_tikhonov(E_obs, r_grid, n_ebins, R0,
                                                lambda_smooth=ls, lambda_norm=ln)
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
            p_r, _, _ = solve_inverse_tikhonov(E_obs, r_grid, n_ebins, R0,
                                                lambda_smooth=ls, lambda_norm=ln)
            psnr = compute_psnr(p_gt, p_r)
            if psnr > best_psnr:
                best_psnr = psnr
                best_params = (ls, ln)
                best_recon = p_r
                best_method = "tikhonov"
    
    print(f"  Tikhonov fine: ls={best_params[0]:.1e}, ln={best_params[1]:.1e} -> PSNR={best_psnr:.2f}")

    # EM / Richardson-Lucy solver
    for n_iter in [200, 500, 1000]:
        for sigma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
            p_r, _, _ = solve_inverse_em(E_obs, r_grid, n_ebins, R0,
                                          n_iter=n_iter, smooth_sigma=sigma)
            psnr = compute_psnr(p_gt, p_r)
            if psnr > best_psnr:
                best_psnr = psnr
                best_recon = p_r
                best_method = f"EM(iter={n_iter},sigma={sigma})"
    
    print(f"  EM best -> PSNR={best_psnr:.2f} ({best_method})")
    print(f"  Final best: {best_method}  ->  PSNR={best_psnr:.2f} dB")

    # Get efficiency histogram for output
    e_edges = np.linspace(0, 1, n_ebins + 1)
    h_counts, _ = np.histogram(E_obs, bins=e_edges)
    h = h_counts.astype(float) / (h_counts.sum() + 1e-12)

    return best_recon, best_params, h, e_edges
