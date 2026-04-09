import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def total_variation(x):
    """Compute Total Variation of image x."""
    dx = np.diff(x, axis=1)
    dy = np.diff(x, axis=0)
    tv = np.sum(np.abs(dx)) + np.sum(np.abs(dy))
    return tv
