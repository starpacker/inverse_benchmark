import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def chambolle_tv_prox(f, weight, n_iter=20):
    """Chambolle's algorithm for TV proximal operator (isotropic TV)."""
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    tau = 0.249
    
    for _ in range(n_iter):
        div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = f + weight * div_p
        gx = np.roll(u, -1, axis=1) - u
        gy = np.roll(u, -1, axis=0) - u
        norm_g = np.sqrt(gx**2 + gy**2 + 1e-16)
        denom = 1.0 + tau * norm_g / weight
        px = (px + tau * gx / weight) / denom
        py = (py + tau * gy / weight) / denom
    
    div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
    return f + weight * div_p
