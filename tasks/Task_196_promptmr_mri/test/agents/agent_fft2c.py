import numpy as np

import matplotlib

matplotlib.use('Agg')

def fft2c(img):
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
