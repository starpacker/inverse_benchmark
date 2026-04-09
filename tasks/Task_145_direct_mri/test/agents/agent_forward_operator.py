import matplotlib

matplotlib.use('Agg')

import numpy as np

def fft2c(img):
    """Centered 2D FFT: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def forward_operator(x, mask):
    """
    MRI Forward operator: image -> undersampled k-space.
    
    Args:
        x: Input image (NxN)
        mask: Undersampling mask (NxN)
    
    Returns:
        y_pred: Predicted undersampled k-space (NxN complex)
    """
    return fft2c(x) * mask
