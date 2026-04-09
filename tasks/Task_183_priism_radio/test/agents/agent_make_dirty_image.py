import numpy as np

import matplotlib

matplotlib.use('Agg')

def make_dirty_image(vis, ui, vi, nx, ny):
    """Create the dirty image (adjoint applied to visibilities), normalized."""
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    psf_grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(psf_grid, (vi, ui), 1.0)
    dirty = np.fft.ifft2(grid).real
    psf = np.fft.ifft2(psf_grid).real
    peak_psf = psf.max()
    if peak_psf > 0:
        dirty /= peak_psf
    return dirty
