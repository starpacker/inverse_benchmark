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

def grid_visibilities(vis, u, v, N, fov):
    """Grid sparse visibilities onto regular UV grid."""
    du = 1.0 / fov
    uv_grid = np.zeros((N, N), dtype=complex)
    weight_grid = np.zeros((N, N))
    for k in range(len(u)):
        iu = int(np.round(u[k] / du)) + N // 2
        iv = int(np.round(v[k] / du)) + N // 2
        if 0 <= iu < N and 0 <= iv < N:
            uv_grid[iv, iu] += vis[k]
            weight_grid[iv, iu] += 1.0
    mask = weight_grid > 0
    uv_grid[mask] /= weight_grid[mask]
    return uv_grid, weight_grid

def make_dirty_image_fft(vis, u, v, N, fov):
    """Compute dirty image via FFT gridding."""
    uv_grid, _ = grid_visibilities(vis, u, v, N, fov)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))).real

def make_dirty_beam_fft(u, v, N, fov):
    """Compute dirty beam via FFT gridding."""
    ones = np.ones(len(u), dtype=complex)
    uv_grid, _ = grid_visibilities(ones, u, v, N, fov)
    beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))).real
    if beam.max() > 0:
        beam /= beam.max()
    return beam

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

def run_inversion(data, clean_gain, clean_niter, clean_thresh, restore_sigma):
    """
    Run the CLEAN inversion algorithm to reconstruct image from visibilities.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data containing:
        - vis_noisy: Noisy visibility measurements
        - u, v: UV coordinates
        - n_pix: Image size
        - fov_uas: Field of view
    clean_gain : float
        CLEAN loop gain
    clean_niter : int
        Maximum CLEAN iterations
    clean_thresh : float
        CLEAN stopping threshold
    restore_sigma : float
        Restoring beam sigma in pixels
        
    Returns
    -------
    dict
        Dictionary containing:
        - cleaned: Final reconstructed image
        - dirty: Dirty image
        - beam: Dirty beam
        - components: CLEAN components
        - residual: Final residual
        - n_clean_iters: Number of CLEAN iterations performed
    """
    vis_noisy = data['vis_noisy']
    u = data['u']
    v = data['v']
    n_pix = data['n_pix']
    fov_uas = data['fov_uas']
    
    # Compute dirty image
    dirty = make_dirty_image_fft(vis_noisy, u, v, n_pix, fov_uas)
    
    # Compute dirty beam
    beam = make_dirty_beam_fft(u, v, n_pix, fov_uas)
    
    # Run CLEAN algorithm
    cleaned, components, residual, n_clean = clean_algorithm(
        dirty, beam, gain=clean_gain, niter=clean_niter,
        threshold=clean_thresh, restore_sigma=restore_sigma
    )
    
    # Enforce non-negativity
    cleaned = np.maximum(cleaned, 0)
    
    return {
        'cleaned': cleaned,
        'dirty': dirty,
        'beam': beam,
        'components': components,
        'residual': residual,
        'n_clean_iters': n_clean
    }
