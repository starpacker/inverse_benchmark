import numpy as np

import matplotlib

matplotlib.use('Agg')

def fft2c(img):
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def forward_operator(x, mask):
    """
    Apply forward operator: FFT then mask.
    
    Parameters:
    -----------
    x : ndarray
        Input image in image domain
    mask : ndarray
        Undersampling mask (boolean or binary)
    
    Returns:
    --------
    y_pred : ndarray
        Undersampled k-space data
    """
    kspace = fft2c(x)
    y_pred = kspace * mask
    return y_pred
