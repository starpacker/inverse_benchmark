import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(C, S):
    """
    Forward model: compute the mixed spectral data matrix.
    
    D = C @ S
    
    Where:
    - C is the concentration matrix (n_samples, n_components)
    - S is the spectra matrix (n_components, n_wavelengths)
    - D is the data matrix (n_samples, n_wavelengths)
    
    Parameters
    ----------
    C : ndarray
        Concentration matrix of shape (n_samples, n_components)
    S : ndarray
        Spectra matrix of shape (n_components, n_wavelengths)
        
    Returns
    -------
    ndarray
        Predicted data matrix D of shape (n_samples, n_wavelengths)
    """
    return C @ S
