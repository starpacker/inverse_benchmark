import numpy as np

def freq_grid(nx, ny, pixel_size):
    """2D spatial frequency magnitude |k| in 1/Å."""
    kx = np.fft.fftfreq(nx, d=pixel_size)
    ky = np.fft.fftfreq(ny, d=pixel_size)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    return np.sqrt(KX**2 + KY**2)
