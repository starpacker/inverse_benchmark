import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.optimize import nnls

def run_inversion(data):
    """
    Recover transfer function Ψ(τ) using Non-Negative Least Squares (NNLS).
    
    This method builds an explicit convolution matrix and solves:
        min ||A @ psi - line_obs||^2  s.t. psi >= 0
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data with keys:
        - continuum: AGN continuum light curve
        - line_obs: observed emission line light curve
        - dt: time step
        - n_time: number of time samples
    
    Returns:
    --------
    dict containing:
        - psi_rec: recovered transfer function
        - ccf_lags: cross-correlation function lag array
        - ccf: cross-correlation function values
    """
    continuum = data['continuum']
    line_obs = data['line_obs']
    dt = data['dt']
    n_time = data['n_time']
    
    # NNLS deconvolution
    M = min(100, n_time)  # number of lag bins to recover
    
    # Build convolution matrix A[i, j] = C(t_i - τ_j) * dt
    A = np.zeros((n_time, M))
    for j in range(M):
        for i in range(j, n_time):
            A[i, j] = continuum[i - j] * dt
    
    # NNLS solve
    psi_nnls, _ = nnls(A, line_obs)
    
    # Embed into full-length array
    psi_rec = np.zeros(n_time)
    psi_rec[:M] = psi_nnls
    
    # Compute cross-correlation function
    c_norm = continuum - continuum.mean()
    l_norm = line_obs - line_obs.mean()
    ccf_full = np.correlate(l_norm, c_norm, mode='full')
    ccf_full = ccf_full / (np.std(continuum) * np.std(line_obs) * n_time)
    lags_full = np.arange(-n_time + 1, n_time) * dt
    
    # Keep only positive lags
    pos_mask = lags_full >= 0
    ccf_lags = lags_full[pos_mask]
    ccf = ccf_full[pos_mask]
    
    return {
        'psi_rec': psi_rec,
        'ccf_lags': ccf_lags,
        'ccf': ccf
    }
