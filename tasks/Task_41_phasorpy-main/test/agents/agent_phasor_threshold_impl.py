import numpy as np

def phasor_threshold_impl(mean, real, imag, mean_min=0):
    """Thresholds phasor coordinates based on intensity."""
    mask = mean < mean_min
    
    mean_out = mean.copy()
    real_out = real.copy()
    imag_out = imag.copy()
    
    mean_out[mask] = np.nan
    real_out[mask] = np.nan
    imag_out[mask] = np.nan
    
    return mean_out, real_out, imag_out
