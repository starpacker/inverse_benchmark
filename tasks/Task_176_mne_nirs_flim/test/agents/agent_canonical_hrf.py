import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

from scipy.stats import gamma as gamma_dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def canonical_hrf(t_hrf, peak=6.0, undershoot=16.0, ratio=6.0):
    """Generate a canonical hemodynamic response function (double-gamma)."""
    h = (gamma_dist.pdf(t_hrf, peak / 1.0, scale=1.0) -
         gamma_dist.pdf(t_hrf, undershoot / 1.0, scale=1.0) / ratio)
    h = h / np.max(np.abs(h))
    return h
