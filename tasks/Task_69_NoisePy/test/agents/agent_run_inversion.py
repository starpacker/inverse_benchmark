import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack as spvstack, diags

from scipy.sparse.linalg import lsqr

def run_inversion(data, forward_data, alpha_values=None):
    """
    Run LSQR tomographic inversion with Laplacian smoothing regularization.
    
    Inverse Problem:
        Minimize ||G @ δs - δt||² + α||L @ δs||²
        where L is the 2D Laplacian smoothing operator.
    
    Args:
        data: dict containing G, nx, ny, c0
        forward_data: dict containing dt_noisy
        alpha_values: list of regularization parameters to try
        
    Returns:
        dict with best reconstruction and inversion results
    """
    if alpha_values is None:
        alpha_values = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    
    G = data['G']
    nx = data['nx']
    ny = data['ny']
    c0 = data['c0']
    dm_gt_flat = data['dm_gt_flat']
    dt_noisy = forward_data['dt_noisy']
    
    # Build 2D Laplacian smoothing matrix
    def diff1d(n):
        return diags([-1, 1], [0, 1], shape=(n - 1, n))
    
    Dx = diff1d(nx)
    Dy = diff1d(ny)
    Lx = spkron(Dx.T @ Dx, speye(ny))
    Ly = spkron(speye(nx), Dy.T @ Dy)
    L = Lx + Ly
    
    damp = 0.01
    best_cc = -1
    best_rec = None
    best_alpha = 1.0
    all_results = []
    
    for alpha in alpha_values:
        # Augmented system for regularization
        G_aug = spvstack([G, np.sqrt(alpha) * L])
        dt_aug = np.concatenate([dt_noisy, np.zeros(L.shape[0])])
        
        # LSQR inversion
        result = lsqr(G_aug, dt_aug, damp=damp, iter_lim=500,
                      atol=1e-8, btol=1e-8)
        ds_rec = result[0]
        
        # Convert back: dm = -c0 * δs
        dm_rec = -c0 * ds_rec
        dm_rec_2d = dm_rec.reshape(nx, ny)
        
        # Compute correlation coefficient
        cc_val = float(np.corrcoef(dm_gt_flat, dm_rec)[0, 1])
        all_results.append({'alpha': alpha, 'cc': cc_val, 'dm_rec_2d': dm_rec_2d})
        
        if cc_val > best_cc:
            best_cc = cc_val
            best_alpha = alpha
            best_rec = dm_rec_2d
    
    return {
        'best_rec': best_rec,
        'best_alpha': best_alpha,
        'best_cc': best_cc,
        'all_results': all_results
    }
