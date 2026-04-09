import numpy as np

from scipy.ndimage import median_filter

def phasor_filter_median_impl(mean, real, imag, size=3, repeat=1):
    """Median filter implementation for phasor coordinates."""
    mean = np.asarray(mean)
    real = np.asarray(real)
    imag = np.asarray(imag)
    
    if repeat == 0:
        return mean, real, imag
        
    for _ in range(repeat):
        real = median_filter(real, size=size)
        imag = median_filter(imag, size=size)
        
    return mean, real, imag
