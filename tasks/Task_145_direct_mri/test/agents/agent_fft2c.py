import matplotlib

matplotlib.use('Agg')

import numpy as np

def fft2c(img):
    """Centered 2D FFT: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
