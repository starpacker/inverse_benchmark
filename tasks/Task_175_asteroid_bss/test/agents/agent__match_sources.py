import numpy as np

import matplotlib

matplotlib.use('Agg')

def _match_sources(gt, est):
    """Match estimated sources to GT via maximum absolute correlation."""
    n_src = gt.shape[0]
    corr_mat = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            corr_mat[i, j] = np.corrcoef(gt[i], est[j])[0, 1]

    # Greedy assignment by |correlation|
    perm = [None] * n_src
    sign = [None] * n_src
    abs_corr = np.abs(corr_mat)
    for _ in range(n_src):
        idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        gt_idx, est_idx = idx
        perm[gt_idx] = est_idx
        sign[gt_idx] = np.sign(corr_mat[gt_idx, est_idx])
        abs_corr[gt_idx, :] = -1
        abs_corr[:, est_idx] = -1

    matched = np.zeros_like(est)
    for i in range(n_src):
        matched[i] = sign[i] * est[perm[i]]
    return matched
