import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_relative_l2(ref, test):
    """Compute relative L2 error."""
    return np.linalg.norm(ref - test) / (np.linalg.norm(ref) + 1e-10)
