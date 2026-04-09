import numpy as np

import matplotlib

matplotlib.use("Agg")

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
