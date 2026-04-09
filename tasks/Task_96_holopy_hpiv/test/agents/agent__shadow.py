import numpy as np

import matplotlib

matplotlib.use("Agg")

def _shadow(nx, ny, dx, x0, y0, r):
    """Generate shadow field for a particle at (x0, y0) with radius r."""
    xx = np.arange(nx) * dx
    yy = np.arange(ny) * dx
    XX, YY = np.meshgrid(xx, yy, indexing="ij")
    s = np.zeros((nx, ny), dtype=complex)
    s[(XX - x0) ** 2 + (YY - y0) ** 2 <= r ** 2] = -1.0
    return s
