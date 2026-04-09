import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.special import voigt_profile

def gaussian(x, amplitude, center, sigma):
    """Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def voigt(x, amplitude, center, sigma, gamma):
    """Voigt peak – convolution of Gaussian and Lorentzian."""
    vp = voigt_profile(x - center, sigma, gamma)
    vp_max = voigt_profile(0.0, sigma, gamma)
    if vp_max > 0:
        return amplitude * vp / vp_max
    return np.zeros_like(x)

def load_and_preprocess_data(seed=42, num_points=2000, x_min=0, x_max=1000, snr_target=30.0):
    """
    Generate synthetic multi-peak spectrum data.
    
    Returns:
        x: numpy array of x-axis values (channel/wavenumber)
        measured_spectrum: numpy array of noisy measured spectrum
        true_peaks: list of dicts with ground-truth peak parameters
        clean_with_baseline: numpy array of clean spectrum with baseline (ground truth)
        baseline_true: numpy array of true baseline
    """
    np.random.seed(seed)
    x = np.linspace(x_min, x_max, num_points)
    
    # Ground-truth peak parameters
    true_peaks = [
        {"type": "gaussian",   "amplitude": 8.0,  "center": 200.0, "sigma": 25.0},
        {"type": "lorentzian", "amplitude": 6.0,  "center": 350.0, "gamma": 20.0},
        {"type": "voigt",      "amplitude": 10.0, "center": 500.0, "sigma": 15.0, "gamma": 10.0},
        {"type": "gaussian",   "amplitude": 5.0,  "center": 650.0, "sigma": 30.0},
        {"type": "lorentzian", "amplitude": 7.0,  "center": 800.0, "gamma": 18.0},
    ]
    
    # Build clean spectrum
    clean_spectrum = np.zeros_like(x)
    for pk in true_peaks:
        if pk["type"] == "gaussian":
            y = gaussian(x, pk["amplitude"], pk["center"], pk["sigma"])
        elif pk["type"] == "lorentzian":
            y = lorentzian(x, pk["amplitude"], pk["center"], pk["gamma"])
        elif pk["type"] == "voigt":
            y = voigt(x, pk["amplitude"], pk["center"], pk["sigma"], pk["gamma"])
        else:
            y = np.zeros_like(x)
        clean_spectrum += y
    
    # Linear baseline
    baseline_true = 0.5 + 0.001 * x
    clean_with_baseline = clean_spectrum + baseline_true
    
    # Add noise (SNR ~ snr_target)
    signal_power = np.mean(clean_spectrum**2)
    noise_power = signal_power / (10 ** (snr_target / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    measured_spectrum = clean_with_baseline + noise
    
    print(f"[DATA] Generating synthetic multi-peak spectrum ...")
    print(f"[DATA]   x range: [{x.min():.0f}, {x.max():.0f}], {len(x)} points")
    print(f"[DATA]   {len(true_peaks)} peaks synthesized")
    print(f"[DATA]   Noise std = {np.sqrt(noise_power):.4f}, target SNR = {snr_target} dB")
    print("[DATA] Done.")
    
    return x, measured_spectrum, true_peaks, clean_with_baseline, baseline_true
