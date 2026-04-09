import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.signal import fftconvolve

def forward_operator(eps_model, wavelet):
    """
    GPR forward operator: compute B-scan from permittivity model.
    
    Physics:
    - Compute reflection coefficients from permittivity contrasts:
      r(z) = (sqrt(ε(z+1)) - sqrt(ε(z))) / (sqrt(ε(z+1)) + sqrt(ε(z)))
    - Convolve reflectivity with source wavelet to get B-scan traces
    
    Parameters:
    -----------
    eps_model : ndarray
        2D permittivity model, shape (nz, nx)
    wavelet : ndarray
        Source wavelet (Ricker), shape (wavelet_pts,)
    
    Returns:
    --------
    bscan : ndarray
        Predicted B-scan data, shape (nz, nx)
    """
    nz, nx = eps_model.shape
    
    # Compute reflectivity for each trace
    reflectivity = np.zeros_like(eps_model)
    for ix in range(nx):
        sqrt_eps = np.sqrt(eps_model[:, ix])
        r = np.zeros(nz)
        r[:-1] = (sqrt_eps[1:] - sqrt_eps[:-1]) / (sqrt_eps[1:] + sqrt_eps[:-1] + 1e-12)
        reflectivity[:, ix] = r
    
    # Convolve each trace with wavelet
    bscan = np.zeros_like(reflectivity)
    for ix in range(nx):
        conv = fftconvolve(reflectivity[:, ix], wavelet, mode='same')
        bscan[:, ix] = conv
    
    return bscan
