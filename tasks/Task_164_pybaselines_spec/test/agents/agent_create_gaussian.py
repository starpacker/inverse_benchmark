import matplotlib

matplotlib.use('Agg')

import os

import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, "repo")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

sys.path.insert(0, REPO_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_gaussian(x, amplitude, center, width):
    """Create a single Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
