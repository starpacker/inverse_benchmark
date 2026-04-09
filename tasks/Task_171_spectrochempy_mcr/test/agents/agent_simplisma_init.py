import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def simplisma_init(D, n_components):
    """
    SIMPLISMA-like initialization using SVD.
    
    Parameters
    ----------
    D : ndarray
        Data matrix (n_samples, n_wavelengths)
    n_components : int
        Number of components to extract
        
    Returns
    -------
    ndarray
        Initial spectra estimate (n_components, n_wavelengths)
    """
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    S_init = np.abs(Vt[:n_components, :])
    # Normalize each spectrum
    for i in range(n_components):
        S_init[i] /= (np.max(S_init[i]) + 1e-12)
    return S_init
