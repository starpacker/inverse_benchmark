import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_psnr(ref, test, data_range=None):
    """Compute Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = max(ref.max() - ref.min(), 1e-10)
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    if mse < 1e-30:
        return 100.0
    return float(10 * np.log10(data_range**2 / mse))
