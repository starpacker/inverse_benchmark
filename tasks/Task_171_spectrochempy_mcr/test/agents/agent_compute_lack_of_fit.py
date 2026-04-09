import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_lack_of_fit(D, D_reconstructed):
    """Compute lack-of-fit percentage."""
    residual = D - D_reconstructed
    lof = np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100
    return lof
