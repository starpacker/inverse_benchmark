import numpy as np

import matplotlib

matplotlib.use('Agg')

def uv_to_grid_indices(u, v, nx, ny):
    """Convert continuous (u,v) to nearest grid indices for FFT grid."""
    ui = np.round(u).astype(int) % nx
    vi = np.round(v).astype(int) % ny
    return ui, vi
