import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_concentration_re(C_true, C_recovered, perm):
    """Compute relative error of recovered concentrations."""
    res = []
    for i, j in enumerate(perm):
        c_true = C_true[:, i]
        c_rec = C_recovered[:, j]
        # Find optimal scaling factor
        scale = np.dot(c_true, c_rec) / (np.dot(c_rec, c_rec) + 1e-12)
        c_rec_scaled = c_rec * scale
        re = np.linalg.norm(c_true - c_rec_scaled) / (np.linalg.norm(c_true) + 1e-12)
        res.append(re)
    return np.array(res)
