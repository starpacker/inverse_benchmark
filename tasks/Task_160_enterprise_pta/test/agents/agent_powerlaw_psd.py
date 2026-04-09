import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def powerlaw_psd(freqs, log10_A, gamma):
    """Power-law power spectral density: S(f) = A^2/(12*pi^2) * (f/f_yr)^(-gamma) * f_yr^(-3)."""
    A = 10.0 ** log10_A
    f_yr = 1.0 / (365.25 * 86400.0)
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs / f_yr) ** (-gamma) * f_yr ** (-3)
