import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import nnls

def csm_nnls_inversion(C, G, alpha=1e-2):
    """
    Direct CSM-based NNLS inversion.
    
    Uses diagonal elements (auto-spectra) and selected off-diagonal elements.
    """
    n_mics = G.shape[0]
    n_grid = G.shape[1]

    A_diag = np.abs(G)**2  # (n_mics, n_grid)
    C_diag = np.real(np.diag(C))

    A_rows = []
    vals = []

    # Diagonal elements
    for i in range(n_mics):
        A_rows.append(A_diag[i, :])
        vals.append(C_diag[i])

    # Off-diagonal: use real parts of upper triangle
    step = max(1, n_mics // 16)
    for i in range(0, n_mics, step):
        for j in range(i + 1, n_mics, step):
            row = np.real(G[i, :] * G[j, :].conj())
            A_rows.append(row)
            vals.append(np.real(C[i, j]))

    A_mat = np.array(A_rows)
    b_vec = np.array(vals)

    # Tikhonov regularization
    A_reg = np.vstack([A_mat, np.sqrt(alpha) * np.eye(n_grid)])
    b_reg = np.concatenate([b_vec, np.zeros(n_grid)])

    q_sol, _ = nnls(A_reg, b_reg)
    return q_sol
