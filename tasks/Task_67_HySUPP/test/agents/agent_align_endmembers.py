import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from itertools import permutations

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def align_endmembers(E_gt, E_rec, A_gt, A_rec):
    """
    Find optimal permutation to align estimated endmembers with GT.
    """
    R = E_gt.shape[1]

    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(R)):
        score = 0
        for i, j in enumerate(perm):
            cos_val = np.dot(E_gt[:, i], E_rec[:, j]) / (
                np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_rec[:, j]) + 1e-12
            )
            score += cos_val
        if score > best_score:
            best_score = score
            best_perm = perm

    perm_list = list(best_perm)
    E_aligned = E_rec[:, perm_list]
    A_aligned = A_rec[perm_list, :]
    return E_aligned, A_aligned, perm_list
