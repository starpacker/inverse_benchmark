import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.special import hankel2

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_95_eispy2d"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def green2d(k0, r):
    """G(r) = (j/4) H_0^{(2)}(k0 r) – 2-D scalar Green's function."""
    r_safe = np.where(r < 1e-12, 1e-12, r)
    return (1j / 4.0) * hankel2(0, k0 * r_safe)
