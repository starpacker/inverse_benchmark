import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.special import hankel2

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_95_eispy2d"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def green2d(k0, r):
    """G(r) = (j/4) H_0^{(2)}(k0 r) – 2-D scalar Green's function."""
    r_safe = np.where(r < 1e-12, 1e-12, r)
    return (1j / 4.0) * hankel2(0, k0 * r_safe)

def forward_operator(chi, k0, gx, gy, tx_angles, rx_pos, ds):
    """
    Forward operator for EM scattering under Born approximation.
    
    Computes the scattered field E_scat = A @ chi where A is the
    sensing matrix based on the Born approximation.
    
    Parameters
    ----------
    chi : ndarray
        Dielectric contrast field (n_grid, n_grid) or flattened
    k0 : float
        Wavenumber
    gx, gy : ndarray
        Grid coordinate vectors
    tx_angles : ndarray
        Transmitter angles
    rx_pos : ndarray
        Receiver positions (n_rx, 2)
    ds : float
        Pixel area
        
    Returns
    -------
    y_pred : ndarray
        Predicted scattered field measurements (n_tx * n_rx,)
    """
    chi_flat = chi.ravel() if chi.ndim > 1 else chi
    
    Xg, Yg = np.meshgrid(gx, gy)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    n_pix = pts.shape[0]
    n_tx = len(tx_angles)
    n_rx = rx_pos.shape[0]
    
    A = np.zeros((n_tx * n_rx, n_pix), dtype=np.complex128)
    
    for l, theta in enumerate(tx_angles):
        d_hat = np.array([np.cos(theta), np.sin(theta)])
        E_inc = np.exp(1j * k0 * (pts @ d_hat))
        
        for m in range(n_rx):
            dr = np.sqrt((rx_pos[m, 0] - pts[:, 0]) ** 2
                         + (rx_pos[m, 1] - pts[:, 1]) ** 2)
            G = green2d(k0, dr)
            A[l * n_rx + m, :] = k0 ** 2 * G * E_inc * ds
    
    y_pred = A @ chi_flat
    return y_pred
