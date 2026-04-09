import numpy as np

import matplotlib

matplotlib.use('Agg')

def ifft2c(kspace):
    """Centered 2D IFFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
