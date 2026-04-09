import numpy as np

from math import pi, sqrt, log10

def angularSpectrum(field, z, wavelength, dx, dy):
    """
    Function to diffract a complex field using the angular spectrum approximation
    Extracted logic from input code.
    """
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * N)
    dfy = 1 / (dy * M)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    # Transfer function
    # Note: Using np.exp for phase
    phase_term = np.exp(1j * z * 2 * pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2)) + 0j))

    tmp = field_spec * phase_term

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out
