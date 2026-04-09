import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def _log_prior(params):
    """Uniform priors on parameters."""
    log10_A_gw, log10_A_red, gamma_red = params
    if not (-18.0 < log10_A_gw < -11.0):
        return -np.inf
    if not (-18.0 < log10_A_red < -11.0):
        return -np.inf
    if not (0.0 < gamma_red < 10.0):
        return -np.inf
    return 0.0
