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
