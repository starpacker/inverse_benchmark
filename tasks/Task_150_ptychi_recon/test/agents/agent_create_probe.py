import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_probe(probe_size):
    """Circular aperture x Gaussian envelope with mild phase curvature."""
    yy, xx = np.mgrid[:probe_size, :probe_size].astype(np.float64)
    cy = cx = probe_size / 2.0
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    R = probe_size * 0.38
    sigma = probe_size * 0.22
    aperture = (r <= R).astype(np.float64)
    gaussian = np.exp(-r**2 / (2 * sigma**2))
    probe = aperture * gaussian * np.exp(-0.3j * (r / R)**2)
    probe /= np.sqrt(np.sum(np.abs(probe)**2))
    return probe
