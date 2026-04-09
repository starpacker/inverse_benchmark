import numpy as np

import matplotlib

matplotlib.use("Agg")

def _asm_kernel(nx, ny, dx, z, wl):
    """Angular Spectrum Method propagation kernel."""
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    kz2 = (1.0 / wl) ** 2 - FX ** 2 - FY ** 2
    prop = kz2 > 0
    return np.exp(1j * 2 * np.pi * np.sqrt(np.maximum(kz2, 0)) * z) * prop
