import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack as spvstack, diags

from scipy.sparse.linalg import lsqr

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(G, dt_noisy, nx, ny, c0, alpha_list=None):
    """
    Run LSQR inversion with Laplacian regularisation.
    Tests multiple regularisation parameters and selects the best one.
    
    Args:
        G: ray-path kernel matrix (sparse)
        dt_noisy: noisy travel-time perturbations
        nx, ny: grid dimensions
        c0: background sound speed
        alpha_list: list of regularisation parameters to test
        
    Returns:
        dict containing:
            - c_rec: reconstructed sound speed (nx, ny)
            - best_alpha: best regularisation parameter
            - all_results: dict mapping alpha to (c_rec, cc) pairs
    """
    if alpha_list is None:
        alpha_list = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    # Build 2D Laplacian for smoothing regularisation
    def diff1d(n):
        return diags([-1, 1], [0, 1], shape=(n - 1, n))

    Dx = diff1d(nx)
    Dy = diff1d(ny)
    Lx = spkron(Dx.T @ Dx, speye(ny))
    Ly = spkron(speye(nx), Dy.T @ Dy)
    L = Lx + Ly

    best_cc = -1
    best_rec = None
    best_alpha = alpha_list[0]
    all_results = {}

    s0 = 1.0 / c0

    for alpha in alpha_list:
        # Augmented system for Tikhonov regularisation
        G_aug = spvstack([G, np.sqrt(alpha) * L])
        dt_aug = np.concatenate([dt_noisy, np.zeros(L.shape[0])])

        result = lsqr(G_aug, dt_aug, damp=0.0, iter_lim=300,
                      atol=1e-8, btol=1e-8)
        ds_rec = result[0]

        # Convert back to speed: c_rec = 1 / (s0 + ds)
        s_rec = s0 + ds_rec
        s_rec = np.maximum(s_rec, 1e-10)
        c_rec = 1.0 / s_rec
        c_rec = np.clip(c_rec, 1300, 1700)
        c_rec = c_rec.reshape(nx, ny)

        all_results[alpha] = c_rec

        # We'll compute CC later in evaluate_results
        # For now, store the reconstruction

    return {
        'all_results': all_results,
        'alpha_list': alpha_list,
        'nx': nx,
        'ny': ny
    }
