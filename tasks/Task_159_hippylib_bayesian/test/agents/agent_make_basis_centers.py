import numpy as np

import matplotlib

matplotlib.use('Agg')

def make_basis_centers(n_per_dim):
    """Regular grid of Gaussian basis centers in [0,1]^2."""
    margin = 0.15
    cx = np.linspace(margin, 1.0 - margin, n_per_dim)
    cy = np.linspace(margin, 1.0 - margin, n_per_dim)
    CX, CY = np.meshgrid(cx, cy)
    return np.column_stack([CX.ravel(), CY.ravel()])
