import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def create_gt_object(n):
    """Create a simple 2D binary object (cross + dots pattern)."""
    obj = np.zeros((n, n), dtype=np.float64)
    c = n // 2

    # Central cross
    obj[c-8:c+8, c-2:c+2] = 1.0
    obj[c-2:c+2, c-8:c+8] = 1.0

    # Corner dots
    for dy, dx in [(-15, -15), (-15, 15), (15, -15), (15, 15)]:
        yy, xx = np.ogrid[:n, :n]
        r = np.sqrt((yy - (c + dy))**2 + (xx - (c + dx))**2)
        obj[r < 4] = 1.0

    # Small rectangle
    obj[c+8:c+14, c-12:c-6] = 0.7

    # Triangle-like shape
    for i in range(8):
        obj[c-18+i, c+6:c+6+2*i+1] = 0.8

    return obj

def estimate_support_from_autocorrelation(measured_magnitude, threshold_fraction=0.08):
    """
    Estimate support from autocorrelation of measured Fourier magnitudes.
    """
    # Power spectrum = measured_magnitude^2
    power_spec = measured_magnitude ** 2
    # Autocorrelation = IFFT of power spectrum
    autocorr = np.real(np.fft.ifft2(power_spec))
    autocorr = np.fft.fftshift(autocorr)

    # Normalize
    autocorr_norm = autocorr / (np.max(autocorr) + 1e-12)

    # Threshold to get support estimate
    support_auto = autocorr_norm > threshold_fraction

    # Clean up with morphological operations
    support_auto = binary_fill_holes(support_auto)
    support_auto = binary_erosion(support_auto, iterations=3)
    support_auto = binary_dilation(support_auto, iterations=2)

    return support_auto

def load_and_preprocess_data(n, noise_level, seed=42):
    """
    Create ground truth object and simulate scattering to produce measured data.
    
    Args:
        n: Image size (n x n)
        noise_level: Noise level on autocorrelation
        seed: Random seed for reproducibility
    
    Returns:
        gt: Ground truth object
        speckle_intensity: Simulated speckle pattern
        measured_magnitude: Measured Fourier magnitude from speckle autocorrelation
        power_spectrum: True power spectrum of the object
        support: Initial support estimate for phase retrieval
    """
    np.random.seed(seed)
    
    # Create ground truth object
    gt = create_gt_object(n)
    
    # Compute Fourier transform of object
    F_obj = np.fft.fft2(gt)

    # Power spectrum = |F[obj]|^2 (this is what autocorrelation gives us)
    power_spectrum = np.abs(F_obj)**2

    # Simulate a random scattering transmission matrix effect
    TM = np.exp(1j * 2 * np.pi * np.random.rand(n, n))
    speckle_field = np.fft.ifft2(F_obj * TM)
    speckle_intensity = np.abs(speckle_field)**2

    # Add noise proportionally to each pixel's power
    noise = noise_level * np.sqrt(np.maximum(power_spectrum, 0)) * np.random.randn(*power_spectrum.shape)
    measured_magnitude = np.sqrt(np.maximum(power_spectrum + noise, 0))
    
    # Estimate support from autocorrelation
    support_auto = estimate_support_from_autocorrelation(measured_magnitude)
    
    # Create a tighter circular support as fallback
    yy, xx = np.ogrid[:n, :n]
    c = n // 2
    r = np.sqrt((yy - c)**2 + (xx - c)**2)
    support_circle = r < 0.2 * n
    
    # Use union of autocorrelation estimate and tight circle
    support = support_auto | support_circle
    
    return gt, speckle_intensity, measured_magnitude, power_spectrum, support
