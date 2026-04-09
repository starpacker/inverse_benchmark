import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_psnr(gt, pred):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-20:
        return 100.0
    max_val = np.max(np.abs(gt))
    return 10.0 * np.log10(max_val**2 / mse)
