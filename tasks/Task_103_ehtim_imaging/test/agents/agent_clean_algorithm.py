import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.signal import fftconvolve

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def clean_algorithm(dirty_image, dirty_beam, gain, niter, threshold, restore_sigma):
    """
    Hogbom CLEAN algorithm — fast slicing implementation.
    
    Parameters
    ----------
    dirty_image : ndarray
        Input dirty image
    dirty_beam : ndarray
        Point spread function (dirty beam)
    gain : float
        CLEAN loop gain
    niter : int
        Maximum number of iterations
    threshold : float
        Stopping threshold (relative to peak)
    restore_sigma : float
        Sigma (in pixels) for the Gaussian restoring beam
        
    Returns
    -------
    tuple
        (restored_image, components, residual, n_iterations)
    """
    N = dirty_image.shape[0]
    residual = dirty_image.copy()
    components = np.zeros_like(dirty_image)
    peak_val = np.abs(residual).max()
    thresh = threshold * peak_val
    bc = N // 2

    for it in range(niter):
        peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak = residual[peak_idx]
        if np.abs(peak) < thresh:
            break
        components[peak_idx] += gain * peak
        sy = peak_idx[0] - bc
        sx = peak_idx[1] - bc
        y1r = max(0, sy)
        y2r = min(N, N + sy)
        x1r = max(0, sx)
        x2r = min(N, N + sx)
        y1b = max(0, -sy)
        y2b = min(N, N - sy)
        x1b = max(0, -sx)
        x2b = min(N, N - sx)
        residual[y1r:y2r, x1r:x2r] -= gain * peak * dirty_beam[y1b:y2b, x1b:x2b]

    # Restore with Gaussian beam
    sigma = restore_sigma
    xx = np.arange(N) - N / 2
    XX, YY = np.meshgrid(xx, xx)
    clean_beam = np.exp(-0.5 * (XX ** 2 + YY ** 2) / sigma ** 2)
    clean_beam /= clean_beam.max()
    restored = fftconvolve(components, clean_beam, mode='same') + residual
    return restored, components, residual, it + 1
