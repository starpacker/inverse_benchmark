import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_relative_l2(gt, pred):
    """Relative L2 error: ||gt - pred||_2 / ||gt||_2"""
    return np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-12)
