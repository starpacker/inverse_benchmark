import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def add_poisson_noise(patterns, photon_count):
    """Scale each pattern to *photon_count* total photons, Poisson-sample."""
    noisy = []
    for pat in patterns:
        s = photon_count / (pat.sum() + 1e-30)
        n = np.random.poisson(np.maximum(pat * s, 0)).astype(np.float64) / s
        noisy.append(n)
    return noisy
