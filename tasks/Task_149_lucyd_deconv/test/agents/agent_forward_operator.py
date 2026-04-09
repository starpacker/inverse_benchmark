import os

import numpy as np

from scipy.signal import fftconvolve

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def forward_operator(x, psf):
    """
    Forward operator for fluorescence microscopy: convolution with PSF.
    
    Parameters
    ----------
    x : ndarray
        Input image (estimate)
    psf : ndarray
        Point spread function
        
    Returns
    -------
    y_pred : ndarray
        Predicted measurement (convolved image)
    """
    y_pred = fftconvolve(x, psf, mode='same')
    y_pred = np.clip(y_pred, 1e-12, None)
    return y_pred
