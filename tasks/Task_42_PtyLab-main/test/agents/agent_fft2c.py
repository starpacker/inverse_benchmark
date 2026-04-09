import scipy.fft

def fft2c(x):
    """Centered 2D FFT."""
    return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(x)))
