import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def project_to_physical(rho):
    """
    Project a Hermitian matrix to the nearest valid density matrix.
    Ensures: Hermitian, positive semi-definite, trace = 1.
    """
    rho = (rho + rho.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    if np.sum(eigenvalues) > 0:
        eigenvalues /= np.sum(eigenvalues)
    rho_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    return rho_physical
