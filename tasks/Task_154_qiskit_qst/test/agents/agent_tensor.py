import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def tensor(*mats):
    """Kronecker product of multiple matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result
