import numpy as np

import scipy.fft

def filter_sinogram(sinogram, window=None):
    """
    Applies the Ram-Lak filter with an optional window function.
    """
    num_angles, num_detectors = sinogram.shape
    
    # Pad to the next power of 2 for efficient FFT
    n = num_detectors
    padded_len = max(64, int(2 ** np.ceil(np.log2(2 * n))))
    
    # Compute frequency axis
    freq = scipy.fft.rfftfreq(padded_len)
    
    # Ram-Lak filter: |f| (ramp filter)
    filt = 2 * np.abs(freq) 
    
    # Apply window
    if window == 'hann':
        w = np.hanning(2 * len(freq))[:len(freq)]
        filt *= w
    elif window == 'hamming':
        w = np.hamming(2 * len(freq))[:len(freq)]
        filt *= w
    
    # Apply filter in Fourier domain
    sino_fft = scipy.fft.rfft(sinogram, n=padded_len, axis=1)
    filtered_sino_fft = sino_fft * filt
    filtered_sino = scipy.fft.irfft(filtered_sino_fft, n=padded_len, axis=1)
    
    # Crop back to original size
    return filtered_sino[:, :num_detectors]
