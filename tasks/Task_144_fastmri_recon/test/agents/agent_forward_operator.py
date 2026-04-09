import matplotlib

matplotlib.use('Agg')

import os

from scipy.fft import fft2, ifft2, fftshift, ifftshift

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(x, mask=None):
    """
    Forward operator: image -> k-space (with optional undersampling).
    
    Applies 2D FFT to convert image to k-space.
    If mask is provided, applies undersampling.
    
    Args:
        x: input image (numpy array)
        mask: optional undersampling mask (numpy array)
    
    Returns:
        kspace: k-space data (complex numpy array)
    """
    kspace_full = fftshift(fft2(ifftshift(x)))
    
    if mask is not None:
        kspace = kspace_full * mask
    else:
        kspace = kspace_full
    
    return kspace
