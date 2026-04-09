import matplotlib

matplotlib.use('Agg')

import numpy as np

def ifft2c(kspace):
    """Centered 2D IFFT: k-space -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
