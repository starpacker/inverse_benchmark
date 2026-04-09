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
