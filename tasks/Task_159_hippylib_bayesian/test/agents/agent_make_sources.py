import numpy as np

import matplotlib

matplotlib.use('Agg')

def make_sources(N_grid, h):
    """Create multiple source terms for multi-experiment inversion."""
    x = np.linspace(h, 1.0 - h, N_grid)
    y = np.linspace(h, 1.0 - h, N_grid)
    X, Y = np.meshgrid(x, y)
    return [
        10.0 * np.ones((N_grid, N_grid)),
        20.0 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * 0.15**2)),
        15.0 * np.exp(-((X - 0.3)**2 + (Y - 0.7)**2) / (2 * 0.12**2)),
        10.0 * np.sin(np.pi * X) * np.sin(np.pi * Y),
    ]
