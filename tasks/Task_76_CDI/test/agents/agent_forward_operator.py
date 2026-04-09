import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fft2, ifft2, fftshift, ifftshift

def forward_operator(obj):
    """
    Compute far-field diffraction intensity pattern.
    I = |FFT(obj)|²
    
    This is the Fraunhofer diffraction forward model.
    
    Args:
        obj: Complex-valued object array
        
    Returns:
        intensity: Diffraction intensity pattern |F{obj}|²
    """
    F_obj = fftshift(fft2(ifftshift(obj)))
    intensity = np.abs(F_obj)**2
    return intensity
