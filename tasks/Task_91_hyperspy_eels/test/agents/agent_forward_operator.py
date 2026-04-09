import matplotlib

matplotlib.use('Agg')

import numpy as np

from numpy.fft import fft, ifft

def forward_operator(x, zlp):
    """
    Forward model: simulate measured EELS with Poisson multiple scattering.
    
    Given the product (t/lambda * S_norm), compute the measured spectrum.
    
    Forward Model:
        F[J] = F[Z] * exp( t/lambda * F[S_norm] )
        
    Since x = t/lambda * S_norm, we have:
        F[J] = F[Z] * exp( F[x] )
    
    Parameters
    ----------
    x : ndarray
        The product t/lambda * S_norm (what we're trying to recover).
    zlp : ndarray
        Zero-loss peak (sum = 1, centered at channel 0).
    
    Returns
    -------
    y_pred : ndarray
        Predicted measured EELS spectrum (non-negative).
    """
    Z = fft(zlp)
    X = fft(x)
    J_ft = Z * np.exp(X)
    y_pred = np.real(ifft(J_ft))
    y_pred = np.maximum(y_pred, 0.0)
    return y_pred
