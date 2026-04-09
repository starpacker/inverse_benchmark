import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.signal import fftconvolve

def forward_operator(volume, psf, photon_count=5000, seed=42):
    """
    Forward model: convolve clean volume with PSF, then add Poisson noise
    to simulate photon-limited fluorescence microscopy acquisition.

    Parameters
    ----------
    volume : np.ndarray
        Clean 3D volume, values in [0, 1].
    psf : np.ndarray
        Point spread function (same shape as volume), normalized.
    photon_count : int
        Peak photon count controlling noise level.
    seed : int
        Random seed.

    Returns
    -------
    blurred_noisy : np.ndarray
        Blurred and noisy observation.
    """
    rng = np.random.RandomState(seed)

    # Convolve with PSF using FFT-based convolution
    blurred = fftconvolve(volume, psf, mode='same')
    blurred = np.clip(blurred, 0, None)

    # Scale to photon counts and apply Poisson noise
    blurred_scaled = blurred * photon_count
    noisy_scaled = rng.poisson(np.clip(blurred_scaled, 0, None)).astype(np.float64)
    blurred_noisy = noisy_scaled / photon_count

    return blurred_noisy
