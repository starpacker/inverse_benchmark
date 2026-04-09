import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def soft_threshold(x, thresh):
    """Complex soft thresholding."""
    mag = np.abs(x)
    return np.where(mag > thresh, x * (1 - thresh / np.maximum(mag, 1e-30)), 0)
