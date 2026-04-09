import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def fourier_design_matrix(toas, n_freq, T):
    """Create the Fourier design matrix F (N_toa x 2*n_freq)."""
    N = len(toas)
    F = np.zeros((N, 2 * n_freq))
    freqs = np.arange(1, n_freq + 1) / T
    for i, f in enumerate(freqs):
        F[:, 2 * i] = np.sin(2.0 * np.pi * f * toas)
        F[:, 2 * i + 1] = np.cos(2.0 * np.pi * f * toas)
    return F, freqs
