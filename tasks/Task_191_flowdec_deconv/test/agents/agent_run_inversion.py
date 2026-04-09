import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.restoration import richardson_lucy

def run_inversion(observed, psf, n_iterations=60):
    """
    Richardson-Lucy deconvolution using scikit-image.

    The RL algorithm iteratively estimates the original image:
        x_{k+1} = x_k * (PSF^T ⊛ (y / (PSF ⊛ x_k)))

    Parameters
    ----------
    observed : np.ndarray
        Blurred + noisy 3D volume.
    psf : np.ndarray
        Point spread function.
    n_iterations : int
        Number of RL iterations.

    Returns
    -------
    deconvolved : np.ndarray
        Reconstructed 3D volume.
    """
    # Ensure non-negative input (RL requires positive values)
    observed_pos = np.clip(observed, 1e-12, None)

    # Extract the compact PSF kernel from the full-size array
    nz, ny, nx = psf.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    # Use a window large enough to capture the PSF
    wz = min(15, nz // 2)
    wy = min(15, ny // 2)
    wx = min(15, nx // 2)
    psf_compact = psf[
        cz - wz:cz + wz + 1,
        cy - wy:cy + wy + 1,
        cx - wx:cx + wx + 1
    ].copy()
    psf_compact /= psf_compact.sum()

    deconvolved = richardson_lucy(observed_pos, psf_compact, num_iter=n_iterations,
                                   clip=False)
    return deconvolved
