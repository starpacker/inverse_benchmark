import numpy as np

import matplotlib

matplotlib.use('Agg')

def clean_sc(C, G, n_iter=500, safety=0.5):
    """CLEAN-SC deconvolution."""
    n_grid = G.shape[1]
    C_rem = C.copy()
    q_clean = np.zeros(n_grid)
    g_norms_sq = np.real(np.sum(G.conj() * G, axis=0))

    for _ in range(n_iter):
        CG = C_rem @ G
        B = np.real(np.sum(G.conj() * CG, axis=0))
        B = np.maximum(B / np.maximum(g_norms_sq**2, 1e-30), 0)

        if np.max(B) < 1e-15:
            break

        j = np.argmax(B)
        g_j = G[:, j:j+1]
        gns = g_norms_sq[j]
        if gns < 1e-30:
            break

        Cg = C_rem @ g_j
        gCg = np.real(g_j.conj().T @ Cg)[0, 0]
        if gCg < 1e-30:
            break

        h = Cg / gCg * gns
        strength = safety * B[j]
        q_clean[j] += strength

        C_rem -= strength * (h @ h.conj().T) / (gns**2)
        C_rem = 0.5 * (C_rem + C_rem.conj().T)

    return q_clean
