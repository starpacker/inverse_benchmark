import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def prox_tv(x, lam, n_iter=10):
    """
    Proximal operator for Total Variation using Chambolle's algorithm.
    """
    ny, nx = x.shape
    p1 = np.zeros((ny, nx), dtype=np.float64)
    p2 = np.zeros((ny, nx), dtype=np.float64)
    tau = 0.25

    for _ in range(n_iter):
        div_p = np.zeros_like(x)
        div_p[:, 1:] += p1[:, 1:]
        div_p[:, 0] += p1[:, 0]
        div_p[:, :-1] -= p1[:, :-1]

        div_p[1:, :] += p2[1:, :]
        div_p[0, :] += p2[0, :]
        div_p[:-1, :] -= p2[:-1, :]

        u = x - lam * div_p
        gx = np.zeros_like(u)
        gy = np.zeros_like(u)
        gx[:, :-1] = u[:, 1:] - u[:, :-1]
        gy[:-1, :] = u[1:, :] - u[:-1, :]

        norm_g = np.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        p1 = (p1 + tau * gx) / (1.0 + tau * norm_g)
        p2 = (p2 + tau * gy) / (1.0 + tau * norm_g)

    div_p = np.zeros_like(x)
    div_p[:, 1:] += p1[:, 1:]
    div_p[:, 0] += p1[:, 0]
    div_p[:, :-1] -= p1[:, :-1]
    div_p[1:, :] += p2[1:, :]
    div_p[0, :] += p2[0, :]
    div_p[:-1, :] -= p2[:-1, :]

    return x - lam * div_p
