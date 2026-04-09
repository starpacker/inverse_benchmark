import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.signal import fftconvolve

from scipy.ndimage import gaussian_filter

def ricker_wavelet(points, a):
    """Ricker wavelet (Mexican hat). Equivalent to scipy.signal.ricker."""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec**2
    mod = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss

def create_gt_permittivity(nz, nx):
    """
    Create a 2D permittivity model with horizontal layers and anomalies.
    Shape: (nz, nx).  Values represent relative permittivity ε_r.
    """
    eps = np.ones((nz, nx)) * 4.0       # background ε_r = 4 (dry sand)

    # Layer 1: top soil (ε_r = 6) from z=30 to z=60
    eps[30:60, :] = 6.0

    # Layer 2: wet clay (ε_r = 15) from z=80 to z=120
    eps[80:120, :] = 15.0

    # Layer 3: bedrock (ε_r = 8) from z=150 onward
    eps[150:, :] = 8.0

    # Anomaly 1: buried pipe (ε_r = 1, air) — circle at (z=45, x=25), r=5
    zz, xx = np.ogrid[:nz, :nx]
    mask1 = (zz - 45)**2 + (xx - 25)**2 < 5**2
    eps[mask1] = 1.0

    # Anomaly 2: water pocket (ε_r = 80) — ellipse at (z=100, x=55)
    mask2 = ((zz - 100) / 6)**2 + ((xx - 55) / 8)**2 < 1
    eps[mask2] = 40.0

    # Anomaly 3: metallic object (ε_r = 30) — small rect at (z=140, x=40)
    eps[137:143, 37:43] = 30.0

    # Smooth slightly to avoid unrealistically sharp transitions
    eps = gaussian_filter(eps, sigma=1.0)
    return eps

def compute_reflection_coefficients(eps_profile):
    """
    From a 1D permittivity profile ε(z), compute reflection coefficients.
    r(z) = (sqrt(ε(z+1)) - sqrt(ε(z))) / (sqrt(ε(z+1)) + sqrt(ε(z)))
    """
    sqrt_eps = np.sqrt(eps_profile)
    r = np.zeros_like(eps_profile)
    r[:-1] = (sqrt_eps[1:] - sqrt_eps[:-1]) / (sqrt_eps[1:] + sqrt_eps[:-1] + 1e-12)
    return r

def load_and_preprocess_data(nz, nx, noise_level, wavelet_pts, wavelet_a, seed):
    """
    Load and preprocess data for GPR Full-Waveform Inversion.
    
    Creates ground truth permittivity model, generates source wavelet,
    computes clean B-scan via forward modeling, and adds noise.
    
    Parameters:
    -----------
    nz : int
        Number of depth samples
    nx : int
        Number of traces (lateral positions)
    noise_level : float
        Fractional noise level (e.g., 0.01 for 1%)
    wavelet_pts : int
        Number of points in Ricker wavelet
    wavelet_a : float
        Ricker wavelet parameter (controls frequency)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    eps_gt : ndarray
        Ground truth permittivity model, shape (nz, nx)
    bscan_noisy : ndarray
        Noisy B-scan data, shape (nz, nx)
    wavelet : ndarray
        Ricker wavelet, shape (wavelet_pts,)
    reflectivity_gt : ndarray
        Ground truth reflectivity, shape (nz, nx)
    """
    np.random.seed(seed)
    
    # Create ground truth permittivity model
    eps_gt = create_gt_permittivity(nz, nx)
    
    # Create source wavelet (Ricker)
    wavelet = ricker_wavelet(wavelet_pts, wavelet_a)
    
    # Compute reflectivity for each trace
    reflectivity_gt = np.zeros_like(eps_gt)
    for ix in range(nx):
        reflectivity_gt[:, ix] = compute_reflection_coefficients(eps_gt[:, ix])
    
    # Convolve each trace with wavelet to get clean B-scan
    bscan_clean = np.zeros_like(reflectivity_gt)
    for ix in range(nx):
        conv = fftconvolve(reflectivity_gt[:, ix], wavelet, mode='same')
        bscan_clean[:, ix] = conv
    
    # Add noise
    amp = np.max(np.abs(bscan_clean)) + 1e-12
    noise = np.random.randn(*bscan_clean.shape) * amp * noise_level
    bscan_noisy = bscan_clean + noise
    
    return eps_gt, bscan_noisy, wavelet, reflectivity_gt
