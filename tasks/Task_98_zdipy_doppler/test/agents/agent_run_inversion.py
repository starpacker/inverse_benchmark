import numpy as np

import matplotlib

matplotlib.use("Agg")

def run_inversion(A, d_noisy, lambda_reg="auto"):
    """
    Tikhonov-regularized inversion to recover surface brightness.

    Solves: min ||A B - d||^2 + lambda ||B - 1||^2

    Parameters
    ----------
    A : ndarray of shape (n_data, n_pix)
        Design matrix
    d_noisy : ndarray of shape (n_data,)
        Noisy observed data
    lambda_reg : float or "auto"
        Regularization parameter

    Returns
    -------
    B_rec : ndarray of shape (n_pix,)
        Reconstructed brightness map (clipped to [0, 1])
    lam_used : float
        The regularization parameter used
    """
    n_pix = A.shape[1]
    ones_vec = np.ones(n_pix)
    d_shifted = d_noisy - A @ ones_vec

    AtA = A.T @ A
    Atd = A.T @ d_shifted

    if lambda_reg == "auto":
        lam_used = np.trace(AtA) / n_pix * 0.3
    else:
        lam_used = float(lambda_reg)

    B_prime = np.linalg.solve(AtA + lam_used * np.eye(n_pix), Atd)
    B_rec = B_prime + ones_vec

    # Clip to physical range
    B_rec = np.clip(B_rec, 0.0, 1.0)

    return B_rec, lam_used
