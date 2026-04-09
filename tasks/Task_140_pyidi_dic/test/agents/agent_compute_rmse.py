import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_rmse(ref, test):
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((ref - test)**2)))
