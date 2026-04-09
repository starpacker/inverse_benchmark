import numpy as np

import matplotlib

matplotlib.use("Agg")

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
