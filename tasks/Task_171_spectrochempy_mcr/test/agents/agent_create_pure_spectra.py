import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def gaussian_peak(wavelengths, center, width, amplitude=1.0):
    """Generate a Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def create_pure_spectra(wavelengths):
    """Create 3 pure component spectra with Gaussian peaks."""
    n_wl = len(wavelengths)
    S = np.zeros((3, n_wl))

    # S1: peaks at 250 nm and 450 nm
    S[0] = gaussian_peak(wavelengths, 250, 20, 1.0) + \
           gaussian_peak(wavelengths, 450, 25, 0.8)

    # S2: peaks at 350 nm and 550 nm
    S[1] = gaussian_peak(wavelengths, 350, 22, 0.9) + \
           gaussian_peak(wavelengths, 550, 28, 1.0)

    # S3: peaks at 300 nm, 500 nm, and 650 nm
    S[2] = gaussian_peak(wavelengths, 300, 18, 0.7) + \
           gaussian_peak(wavelengths, 500, 20, 0.85) + \
           gaussian_peak(wavelengths, 650, 22, 0.6)

    return S
