import matplotlib

matplotlib.use('Agg')

import numpy as np

def compute_tv_gradient(x):
    """Compute the gradient of an isotropic TV approximation."""
    eps = 1e-8
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    grad_mag = np.sqrt(dx ** 2 + dy ** 2 + eps)
    nx = dx / grad_mag
    ny = dy / grad_mag
    div_x = nx - np.roll(nx, 1, axis=1)
    div_y = ny - np.roll(ny, 1, axis=0)
    return -(div_x + div_y)
