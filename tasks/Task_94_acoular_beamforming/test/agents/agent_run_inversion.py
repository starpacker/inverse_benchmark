import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import nnls

from scipy.ndimage import gaussian_filter

def remove_csm_diagonal(C):
    """Set CSM diagonal to zero to remove uncorrelated noise."""
    C_clean = C.copy()
    np.fill_diagonal(C_clean, 0)
    return C_clean

def conventional_beamforming(C, G):
    """Vectorized delay-and-sum beamforming."""
    CG = C @ G
    B = np.real(np.sum(G.conj() * CG, axis=0))
    norms = np.real(np.sum(G.conj() * G, axis=0))
    norms_sq = np.maximum(norms**2, 1e-30)
    return np.maximum(B / norms_sq, 0)

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

def run_inversion(C, G, grid_res, clean_iterations=500, clean_safety=0.5, nnls_alpha=1e-2):
    """
    Run inversion to recover source distribution from CSM.
    
    Implements three methods:
    1. Conventional Beamforming (Delay-and-Sum)
    2. CLEAN-SC deconvolution with Gaussian smoothing
    3. Direct CSM-based NNLS inversion
    
    Parameters
    ----------
    C : ndarray
        Cross-spectral matrix (n_mics, n_mics)
    G : ndarray
        Steering vector matrix (n_mics, n_grid)
    grid_res : int
        Grid resolution (grid is grid_res x grid_res)
    clean_iterations : int
        Number of CLEAN-SC iterations
    clean_safety : float
        CLEAN-SC safety factor (0 < safety <= 1)
    nnls_alpha : float
        Tikhonov regularization parameter for NNLS
        
    Returns
    -------
    dict
        Dictionary containing reconstructed source maps:
        - conventional: (n_grid,) conventional beamforming result
        - clean_sc: (n_grid,) CLEAN-SC result (smoothed)
        - nnls: (n_grid,) NNLS result (smoothed)
    """
    # Remove CSM diagonal for noise-free beamforming
    C_clean = remove_csm_diagonal(C)
    
    # 1. Conventional Beamforming
    B_conv = conventional_beamforming(C_clean, G)
    
    # 2. CLEAN-SC with Gaussian smoothing
    q_clean_raw = clean_sc(C_clean, G, n_iter=clean_iterations, safety=clean_safety)
    q_clean = gaussian_filter(q_clean_raw.reshape(grid_res, grid_res), sigma=1.5).ravel()
    
    # 3. NNLS inversion with Gaussian smoothing
    q_nnls_raw = csm_nnls_inversion(C_clean, G, alpha=nnls_alpha)
    q_nnls = gaussian_filter(q_nnls_raw.reshape(grid_res, grid_res), sigma=1.0).ravel()
    
    return {
        'conventional': B_conv,
        'clean_sc': q_clean,
        'nnls': q_nnls
    }
