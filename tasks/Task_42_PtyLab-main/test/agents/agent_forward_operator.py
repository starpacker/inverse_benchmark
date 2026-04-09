import scipy.fft

def fft2c(x):
    """Centered 2D FFT."""
    return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(x)))

def forward_operator(object_patch, probe):
    """
    Performs the physical forward model: Exit Wave -> FFT.
    
    Args:
        object_patch (np.array): Complex object patch.
        probe (np.array): Complex probe.
        
    Returns:
        np.array: The predicted far-field complex wave (before magnitude).
    """
    exit_wave = object_patch * probe
    wave_fourier = fft2c(exit_wave)
    return wave_fourier
