import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.optimize import nnls

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

def mcr_als_manual(D, n_components, max_iter=500, tol=1e-6):
    """
    Manual MCR-ALS implementation with non-negativity constraints.

    Algorithm:
        1. Initialize S via SVD
        2. Repeat until convergence:
            a. Fix S, solve for C using NNLS (column-wise)
            b. Fix C, solve for S using NNLS (column-wise)
            c. Check convergence (relative change in lack-of-fit)
            
    Parameters
    ----------
    D : ndarray
        Data matrix (n_samples, n_wavelengths)
    n_components : int
        Number of components
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    tuple
        (C, S, lof_history) - concentration matrix, spectra matrix, and LOF history
    """
    n_samples, n_wavelengths = D.shape

    # Initialize spectra via SVD
    S = simplisma_init(D, n_components)

    lof_prev = np.inf
    lof_history = []

    for iteration in range(max_iter):
        # Step A: Fix S, solve for C using NNLS
        C = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            C[i, :], _ = nnls(S.T, D[i, :])

        # Step B: Fix C, solve for S using NNLS
        S_new = np.zeros((n_components, n_wavelengths))
        for j in range(n_wavelengths):
            S_new[:, j], _ = nnls(C, D[:, j])
        S = S_new

        # Compute lack-of-fit using forward operator
        D_reconstructed = forward_operator(C, S)
        residual = D - D_reconstructed
        lof = np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100

        lof_history.append(lof)

        # Check convergence
        if abs(lof_prev - lof) < tol:
            print(f"MCR-ALS converged at iteration {iteration + 1}, LOF = {lof:.4f}%")
            break
        lof_prev = lof

    else:
        print(f"MCR-ALS reached max iterations ({max_iter}), LOF = {lof:.4f}%")

    return C, S, lof_history
