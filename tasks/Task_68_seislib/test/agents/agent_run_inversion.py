import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack

from scipy.sparse.linalg import lsqr

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(G, dt_noisy, nx, ny, alpha_values=None):
    """
    Run LSQR inversion with Laplacian regularization.
    
    Solves the augmented system:
        [G; sqrt(α) L] @ δm = [δt; 0]
    
    Args:
        G: sparse kernel matrix (n_rays, n_cells)
        dt_noisy: noisy travel-time residuals (n_rays,)
        nx, ny: grid dimensions
        alpha_values: list of regularization strengths to try
    
    Returns:
        dict containing:
            - dm_rec: best reconstructed perturbation (nx, ny)
            - dm_rec_flat: flattened reconstruction
            - best_alpha: optimal regularization strength
            - best_cc: correlation coefficient for best result
            - all_results: dict mapping alpha to (dm_rec, cc) tuples
    """
    if alpha_values is None:
        alpha_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    # Build 2D Laplacian for regularization
    def diff1d(n):
        D = np.zeros((n, n))
        for i in range(n - 1):
            D[i, i] = -1
            D[i, i + 1] = 1
        return csr_matrix(D)
    
    Dx = diff1d(nx)
    Dy = diff1d(ny)
    Ix = speye(nx)
    Iy = speye(ny)
    
    Lx = spkron(Dx.T @ Dx, Iy)
    Ly = spkron(Ix, Dy.T @ Dy)
    L = Lx + Ly
    
    n_cells = nx * ny
    all_results = {}
    best_cc = -1.0
    best_alpha = alpha_values[0]
    best_rec = None
    
    for alpha in alpha_values:
        # Augmented system
        G_aug = vstack([G, np.sqrt(alpha) * L])
        dt_aug = np.concatenate([dt_noisy, np.zeros(L.shape[0])])
        
        result = lsqr(G_aug, dt_aug, damp=0.0, iter_lim=500, atol=1e-8, btol=1e-8)
        dm_rec_flat = result[0]
        dm_rec_2d = dm_rec_flat.reshape(nx, ny)
        
        all_results[alpha] = {
            'dm_rec': dm_rec_2d,
            'dm_rec_flat': dm_rec_flat
        }
    
    return {
        'all_results': all_results,
        'L': L,
        'nx': nx,
        'ny': ny
    }
