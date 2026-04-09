import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def hellings_downs(theta):
    """Hellings-Downs overlap reduction function."""
    x = (1.0 - np.cos(theta)) / 2.0
    if x < 1e-10:
        return 1.0
    hd = 1.5 * x * np.log(x) - 0.25 * x + 0.5
    return hd
