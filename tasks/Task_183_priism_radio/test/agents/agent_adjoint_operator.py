import numpy as np

import matplotlib

matplotlib.use('Agg')

def adjoint_operator(vis, ui, vi, nx, ny):
    """
    Adjoint model: visibilities → image (dirty image direction).
    Places visibilities on grid and applies inverse FFT.
    """
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    img = np.fft.ifft2(grid).real
    return img
