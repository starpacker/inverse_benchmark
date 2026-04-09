import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_101_pyilc_cmb"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def run_inversion(data, freqs_ghz):
    """
    Apply Internal Linear Combination (ILC) to recover CMB.
    
    ILC weights: w = (a^T C^{-1} a)^{-1} C^{-1} a
    where a = [1, 1, ..., 1] (CMB has unit response at all frequencies).
    
    Recovered CMB = w^T d
    
    Parameters
    ----------
    data : np.ndarray, shape (n_freq, n_pix, n_pix)
        Multi-frequency observed maps.
    freqs_ghz : np.ndarray
        Observation frequencies (for reference).
    
    Returns
    -------
    cmb_rec : np.ndarray, shape (n_pix, n_pix)
        Recovered CMB map.
    weights : np.ndarray, shape (n_freq,)
        ILC weights.
    """
    n_freq, ny, nx = data.shape
    
    # Flatten spatial dimensions
    data_flat = data.reshape(n_freq, -1)
    
    # CMB response vector (unit response in thermodynamic units)
    a = np.ones(n_freq)
    
    # Compute covariance matrix across frequencies
    C = np.cov(data_flat)
    
    # Regularize for numerical stability
    C += 1e-10 * np.eye(n_freq)
    
    # Invert covariance
    C_inv = np.linalg.inv(C)
    
    # Compute ILC weights: w = (a^T C^{-1} a)^{-1} C^{-1} a
    weights = C_inv @ a / (a @ C_inv @ a)
    
    # Apply weights to recover CMB
    cmb_flat = weights @ data_flat
    cmb_rec = cmb_flat.reshape(ny, nx)
    
    return cmb_rec, weights
