import numpy as np

import matplotlib

matplotlib.use("Agg")

def forward_operator(obj_patch, probe):
    """
    Single-position forward model for ptychography.
    
    Computes the diffraction pattern intensity from an object patch and probe.
    
    Forward model: exit_wave = probe * object_patch → I = |FFT(exit_wave)|²
    
    Parameters
    ----------
    obj_patch : ndarray (complex)
        Complex object patch at the probe position
    probe : ndarray (complex)
        Complex probe function
    
    Returns
    -------
    intensity : ndarray (float)
        Predicted diffraction pattern intensity |FFT(probe * obj_patch)|²
    """
    exit_wave = obj_patch * probe
    fourier_wave = np.fft.fft2(exit_wave)
    intensity = np.abs(fourier_wave) ** 2
    return intensity
