import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(image, u, v, fov):
    """
    Forward operator: compute visibilities from image via NDFT.
    
    Implements V(u,v) = FFT[I(x,y)] sampled at sparse uv-points.
    
    Parameters
    ----------
    image : ndarray
        Input image (N x N)
    u : ndarray
        U coordinates of visibility points
    v : ndarray
        V coordinates of visibility points
    fov : float
        Field of view in micro-arcseconds
        
    Returns
    -------
    ndarray
        Complex visibility values at (u, v) points
    """
    N = image.shape[0]
    pix_size = fov / N
    x = (np.arange(N) - N / 2) * pix_size
    y = (np.arange(N) - N / 2) * pix_size
    X, Y = np.meshgrid(x, y)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    img_flat = image.ravel()
    n_vis = len(u)
    vis = np.zeros(n_vis, dtype=complex)
    batch = 200
    for start in range(0, n_vis, batch):
        end = min(start + batch, n_vis)
        phase = -2.0 * np.pi * (np.outer(u[start:end], x_flat) + np.outer(v[start:end], y_flat))
        vis[start:end] = np.dot(np.exp(1j * phase), img_flat)
    return vis
