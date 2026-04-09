import matplotlib

matplotlib.use('Agg')

import os

import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, "repo")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

sys.path.insert(0, REPO_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_gaussian(x, amplitude, center, width):
    """Create a single Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def load_and_preprocess_data(n_points=2000, seed=42):
    """
    Synthesize a spectrum with known signal, baseline, and noise.
    
    Parameters
    ----------
    n_points : int
        Number of spectral points.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'x': wavenumber axis (1D array)
        - 'measured': measured spectrum (1D array)
        - 'true_signal': ground truth signal (1D array)
        - 'true_baseline': ground truth baseline (1D array)
        - 'noise': noise component (1D array)
    """
    print("[DATA] Synthesizing spectral data...")
    rng = np.random.default_rng(seed)

    # Wavenumber axis (e.g., Raman: 200-3500 cm^-1)
    x = np.linspace(200, 3500, n_points)
    x_norm = (x - x.min()) / (x.max() - x.min())  # normalize to [0,1]

    # ── True signal: multiple Gaussian peaks ──
    peaks = [
        {'amplitude': 1.0,  'center': 520,  'width': 25},
        {'amplitude': 0.7,  'center': 785,  'width': 15},
        {'amplitude': 1.5,  'center': 1100, 'width': 40},
        {'amplitude': 0.5,  'center': 1350, 'width': 20},
        {'amplitude': 1.2,  'center': 1580, 'width': 30},
        {'amplitude': 0.8,  'center': 2100, 'width': 50},
        {'amplitude': 0.6,  'center': 2450, 'width': 35},
        {'amplitude': 1.8,  'center': 2920, 'width': 45},
        {'amplitude': 0.4,  'center': 3200, 'width': 20},
    ]
    true_signal = np.zeros_like(x)
    for p in peaks:
        true_signal += create_gaussian(x, p['amplitude'], p['center'], p['width'])
    print(f"[DATA]   Created {len(peaks)} Gaussian peaks, signal range: "
          f"[{true_signal.min():.3f}, {true_signal.max():.3f}]")

    # ── True baseline: 4th order polynomial + broad Gaussian hump ──
    coeffs = [0.3, -0.8, 1.2, -0.5, 0.2]  # polynomial coefficients
    true_baseline = np.polyval(coeffs, x_norm)
    # Add a broad fluorescence-like hump
    true_baseline += 0.5 * create_gaussian(x, 1.0, 1800, 600)
    print(f"[DATA]   Baseline range: [{true_baseline.min():.3f}, {true_baseline.max():.3f}]")

    # ── Noise ──
    noise_level = 0.03
    noise = rng.normal(0, noise_level, n_points)

    # ── Forward model: measured = signal + baseline + noise ──
    measured = true_signal + true_baseline + noise
    print(f"[DATA]   Measured spectrum range: [{measured.min():.3f}, {measured.max():.3f}]")
    print(f"[DATA]   SNR ~ {np.std(true_signal) / noise_level:.1f}")
    print(f"[DATA] Data synthesis complete. {n_points} points.")

    data_dict = {
        'x': x,
        'measured': measured,
        'true_signal': true_signal,
        'true_baseline': true_baseline,
        'noise': noise
    }
    
    return data_dict
