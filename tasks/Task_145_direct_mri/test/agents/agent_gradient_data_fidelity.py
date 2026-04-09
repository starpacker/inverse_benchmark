import matplotlib

matplotlib.use('Agg')

import numpy as np

def fft2c(img):
    """Centered 2D FFT: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    """Centered 2D IFFT: k-space -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

def gradient_data_fidelity(x, kspace_under, mask):
    """
    Gradient of data fidelity term: ||MFx - y||^2
    grad = F^H M^H (MFx - y) = F^H M (MFx - y)
    """
    Fx = fft2c(x)
    residual = mask * Fx - kspace_under
    grad = ifft2c(mask * residual)
    return np.real(grad)
