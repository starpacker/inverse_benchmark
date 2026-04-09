import scipy.fft

def ifft2c(x):
    """Centered 2D Inverse FFT."""
    return scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.ifftshift(x)))
