import numpy as np

import matplotlib

matplotlib.use('Agg')

def divergence_2d(gx, gy):
    """Compute divergence (adjoint of gradient)."""
    dx = np.zeros_like(gx)
    dy = np.zeros_like(gy)
    dx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
    dx[:, 0] = gx[:, 0]
    dx[:, -1] = -gx[:, -2]
    dy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
    dy[0, :] = gy[0, :]
    dy[-1, :] = -gy[-2, :]
    return dx + dy
