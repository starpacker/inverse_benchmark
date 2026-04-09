import matplotlib

matplotlib.use('Agg')

from numpy.linalg import lstsq, norm

def run_inversion(I_noisy, W, normalize=True):
    """
    Recover Mueller matrix from noisy intensity measurements via least squares.
    
    Solves: min_m ||W @ m - I_noisy||^2
    
    Parameters
    ----------
    I_noisy : ndarray, shape (N,)
        Noisy intensity measurements.
    W : ndarray, shape (N, 16)
        Measurement matrix.
    normalize : bool
        If True, normalize so M[0,0] = 1.
    
    Returns
    -------
    M_recon : ndarray, shape (4, 4)
        Reconstructed Mueller matrix.
    """
    m_vec, residuals, rank, sv = lstsq(W, I_noisy, rcond=None)
    M_recon = m_vec.reshape(4, 4)
    
    if normalize and abs(M_recon[0, 0]) > 1e-10:
        M_recon = M_recon / M_recon[0, 0]
    
    return M_recon
