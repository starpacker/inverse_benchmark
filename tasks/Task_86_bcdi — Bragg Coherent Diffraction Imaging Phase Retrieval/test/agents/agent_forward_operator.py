import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def forward_operator(obj):
    """
    Forward operator: Compute far-field diffraction intensity.
    I(q) = |FFT{ρ(r)}|²
    
    In Bragg CDI, the measurement is the 3D diffraction intensity
    around a Bragg peak.
    
    Args:
        obj: 3D complex object (electron density with phase)
        
    Returns:
        intensity: 3D diffraction intensity pattern
    """
    obj_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    intensity = np.abs(obj_ft)**2
    return intensity
