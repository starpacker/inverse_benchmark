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
