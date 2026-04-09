import numpy as np

import matplotlib

matplotlib.use('Agg')

def gradient_2d(img):
    """Compute discrete gradient (finite differences)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy

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

def tv_prox(img, lam, n_inner=80):
    """
    Proximal operator for isotropic TV using Chambolle's projection algorithm.
    Solves: argmin_x 0.5*||x - img||^2 + lam*TV(x)
    """
    if lam <= 0:
        return img.copy()

    px = np.zeros_like(img)
    py = np.zeros_like(img)
    tau = 1.0 / 8.0

    for _ in range(n_inner):
        div_p = divergence_2d(px, py)
        gx, gy = gradient_2d(div_p - img / lam)
        px_new = px + tau * gx
        py_new = py + tau * gy
        norm = np.sqrt(px_new**2 + py_new**2)
        norm = np.maximum(norm, 1.0)
        px = px_new / norm
        py = py_new / norm

    return img - lam * divergence_2d(px, py)
