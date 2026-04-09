import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def match_components(S_true, S_recovered):
    """
    Match recovered components to true components using correlation.
    Returns permutation indices and correlation matrix.
    """
    n_comp = S_true.shape[0]
    corr_matrix = np.zeros((n_comp, n_comp))

    for i in range(n_comp):
        for j in range(n_comp):
            s_true_norm = S_true[i] / (np.linalg.norm(S_true[i]) + 1e-12)
            s_rec_norm = S_recovered[j] / (np.linalg.norm(S_recovered[j]) + 1e-12)
            corr_matrix[i, j] = np.abs(np.dot(s_true_norm, s_rec_norm))

    # Greedy matching
    perm = []
    used = set()
    for i in range(n_comp):
        best_j = -1
        best_corr = -1
        for j in range(n_comp):
            if j not in used and corr_matrix[i, j] > best_corr:
                best_corr = corr_matrix[i, j]
                best_j = j
        perm.append(best_j)
        used.add(best_j)

    return perm, corr_matrix
