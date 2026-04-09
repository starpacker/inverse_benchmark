import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_psnr(D_true, D_reconstructed):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((D_true - D_reconstructed) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(D_true))
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr
