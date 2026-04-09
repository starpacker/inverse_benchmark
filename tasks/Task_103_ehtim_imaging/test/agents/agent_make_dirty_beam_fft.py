import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def grid_visibilities(vis, u, v, N, fov):
    """Grid sparse visibilities onto regular UV grid."""
    du = 1.0 / fov
    uv_grid = np.zeros((N, N), dtype=complex)
    weight_grid = np.zeros((N, N))
    for k in range(len(u)):
        iu = int(np.round(u[k] / du)) + N // 2
        iv = int(np.round(v[k] / du)) + N // 2
        if 0 <= iu < N and 0 <= iv < N:
            uv_grid[iv, iu] += vis[k]
            weight_grid[iv, iu] += 1.0
    mask = weight_grid > 0
    uv_grid[mask] /= weight_grid[mask]
    return uv_grid, weight_grid

def make_dirty_beam_fft(u, v, N, fov):
    """Compute dirty beam via FFT gridding."""
    ones = np.ones(len(u), dtype=complex)
    uv_grid, _ = grid_visibilities(ones, u, v, N, fov)
    beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))).real
    if beam.max() > 0:
        beam /= beam.max()
    return beam
