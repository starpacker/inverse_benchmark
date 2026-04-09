import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_spectral_cc(S_true, S_recovered, perm):
    """Compute correlation coefficients between true and recovered spectra."""
    ccs = []
    for i, j in enumerate(perm):
        cc = np.corrcoef(S_true[i], S_recovered[j])[0, 1]
        ccs.append(abs(cc))
    return np.array(ccs)
