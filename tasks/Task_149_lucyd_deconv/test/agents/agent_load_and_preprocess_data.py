import os

import numpy as np

from scipy.signal import fftconvolve

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def load_and_preprocess_data(size=128, psf_size=21, psf_sigma=2.5, photon_gain=500):
    """
    Generate synthetic fluorescence microscopy data and PSF.
    
    Returns
    -------
    ground_truth : ndarray
        Clean ground truth image
    psf : ndarray
        Point spread function
    blurred_noisy : ndarray
        Observed image (blurred + Poisson noise)
    params : dict
        Parameters used for data generation
    """
    # Create synthetic microscopy image
    image = np.zeros((size, size), dtype=np.float64)

    # --- Fluorescent beads (point sources) ---
    n_beads = 30
    for _ in range(n_beads):
        x = np.random.randint(10, size - 10)
        y = np.random.randint(10, size - 10)
        brightness = np.random.uniform(0.5, 1.0)
        sigma_bead = np.random.uniform(0.8, 1.5)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        bead = brightness * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma_bead**2))
        image += bead

    # --- Cell-like elliptical structures ---
    n_cells = 3
    for _ in range(n_cells):
        cx = np.random.randint(25, size - 25)
        cy = np.random.randint(25, size - 25)
        a = np.random.uniform(8, 18)
        b = np.random.uniform(5, 12)
        theta = np.random.uniform(0, np.pi)
        brightness = np.random.uniform(0.2, 0.5)

        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        dx = xx - cx
        dy = yy - cy
        xr = dx * np.cos(theta) + dy * np.sin(theta)
        yr = -dx * np.sin(theta) + dy * np.cos(theta)
        ellipse = np.exp(-((xr / a)**2 + (yr / b)**2))
        ring = ellipse * (1 - 0.7 * np.exp(-((xr / (a * 0.6))**2 + (yr / (b * 0.6))**2)))
        image += brightness * ring

    # --- Filamentous structures (microtubules) ---
    n_filaments = 5
    for _ in range(n_filaments):
        x0, y0 = np.random.randint(5, size - 5, 2)
        angle = np.random.uniform(0, np.pi)
        length = np.random.uniform(30, 60)
        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)
        brightness = np.random.uniform(0.3, 0.6)
        width = np.random.uniform(1.0, 2.0)

        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        dx_line = x1 - x0
        dy_line = y1 - y0
        seg_len = np.sqrt(dx_line**2 + dy_line**2) + 1e-12
        t = np.clip(((xx - x0) * dx_line + (yy - y0) * dy_line) / (seg_len**2), 0, 1)
        px = x0 + t * dx_line
        py = y0 + t * dy_line
        dist = np.sqrt((xx - px)**2 + (yy - py)**2)
        filament = brightness * np.exp(-dist**2 / (2 * width**2))
        image += filament

    # Normalize to [0, 1]
    ground_truth = image / (image.max() + 1e-12)

    # Create Gaussian PSF
    ax = np.arange(psf_size) - psf_size // 2
    yy_psf, xx_psf = np.meshgrid(ax, ax, indexing='ij')
    psf = np.exp(-(xx_psf**2 + yy_psf**2) / (2 * psf_sigma**2))
    psf /= psf.sum()

    # Apply forward model: convolution + Poisson noise
    blurred = fftconvolve(ground_truth, psf, mode='same')
    blurred = np.clip(blurred, 0, None)
    blurred_noisy = np.random.poisson(blurred * photon_gain).astype(np.float64) / photon_gain

    params = {
        'size': size,
        'psf_size': psf_size,
        'psf_sigma': psf_sigma,
        'photon_gain': photon_gain
    }

    return ground_truth, psf, blurred_noisy, params
