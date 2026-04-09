import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy import sparse

from scipy.sparse.linalg import lsqr

def _build_laplacian_2d(nr, nz):
    """Build a sparse 2D Laplacian operator for an (nr, nz) grid."""
    n = nr * nz
    diags = []
    offsets = []

    # Main diagonal: -4 (or fewer at boundaries handled by adjacency)
    main = -4.0 * np.ones(n)
    diags.append(main)
    offsets.append(0)

    # Right neighbour (+1 in z direction)
    d = np.ones(n - 1)
    # Zero out wrap-around at z boundaries
    for i in range(n - 1):
        if (i + 1) % nz == 0:
            d[i] = 0.0
    diags.append(d)
    offsets.append(1)

    # Left neighbour (-1 in z direction)
    d = np.ones(n - 1)
    for i in range(n - 1):
        if (i + 1) % nz == 0:
            d[i] = 0.0
    diags.append(d)
    offsets.append(-1)

    # Down neighbour (+nz in r direction)
    diags.append(np.ones(n - nz))
    offsets.append(nz)

    # Up neighbour (-nz in r direction)
    diags.append(np.ones(n - nz))
    offsets.append(-nz)

    Lap = sparse.diags(diags, offsets, shape=(n, n), format="csr")
    return Lap

def run_inversion(measurements, geometry_matrix, nr, nz, tikhonov_lambda, lsqr_iter_limit):
    """
    Run the Tikhonov-regularized LSQR reconstruction.
    
    Solves: min_x || L x - y ||^2 + lambda * || D x ||^2
    where D is the 2D Laplacian (smoothness prior).
    
    Parameters
    ----------
    measurements : ndarray
        Noisy line-integrated measurements (n_los,)
    geometry_matrix : sparse matrix
        The geometry matrix L
    nr : int
        Number of R grid points
    nz : int
        Number of Z grid points
    tikhonov_lambda : float
        Regularization weight
    lsqr_iter_limit : int
        Maximum LSQR iterations
        
    Returns
    -------
    recon_2d : ndarray
        Reconstructed emissivity field of shape (nr, nz)
    """
    print("[4/6] Tikhonov-regularised LSQR reconstruction …")
    
    L = geometry_matrix
    y = measurements
    
    # Build 2D Laplacian regularization matrix
    D = _build_laplacian_2d(nr, nz)
    
    # Stack: [L; sqrt(lam)*D]
    A = sparse.vstack([L, np.sqrt(tikhonov_lambda) * D], format="csr")
    b = np.concatenate([y, np.zeros(D.shape[0])])
    
    # Solve via LSQR
    result = lsqr(A, b, iter_lim=lsqr_iter_limit, atol=1e-12, btol=1e-12)
    x_hat = result[0]
    
    # Enforce non-negativity (emissivity ≥ 0)
    x_hat = np.clip(x_hat, 0, None)
    
    # Reshape to 2D
    recon_2d = x_hat.reshape(nr, nz)
    
    return recon_2d
