import matplotlib

matplotlib.use('Agg')

import numpy as np

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def gaussian_peak(x, center, width, height):
    """Single Gaussian peak."""
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)

def lorentzian_peak(x, center, width, height):
    """Single Lorentzian peak."""
    return height / (1 + ((x - center) / width) ** 2)

def load_and_preprocess_data(n_points=500, n_mixtures=15, snr=25.0, seed=42):
    """
    Synthesize and preprocess Raman spectral data.
    
    Creates pure component spectra (3 minerals), generates mixed spectra with
    random mixing proportions, adds polynomial baseline drift and Gaussian noise.
    
    Parameters
    ----------
    n_points : int
        Number of wavenumber points.
    n_mixtures : int
        Number of mixed spectra to generate.
    snr : float
        Signal-to-noise ratio.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    data : dict
        Dictionary containing:
        - wavenumber: array of wavenumber values
        - pure_components: (3, n_points) array of pure component spectra
        - true_weights: (n_mixtures, 3) array of true mixing proportions
        - clean_spectra: (n_mixtures, n_points) array of clean mixed spectra
        - baselines: (n_mixtures, n_points) array of true baselines
        - observed_spectra: (n_mixtures, n_points) array of observed spectra with noise
        - n_components: number of components
    """
    np.random.seed(seed)
    
    # Wavenumber axis
    wavenumber = np.linspace(200, 1200, n_points)
    
    # --- Pure component spectra ---
    # Component 1: Mineral A – peaks at 400, 600 cm⁻¹
    comp1 = (gaussian_peak(wavenumber, 400, 20, 1.0) +
             lorentzian_peak(wavenumber, 600, 15, 0.7))
    
    # Component 2: Mineral B – peaks at 500, 800 cm⁻¹
    comp2 = (gaussian_peak(wavenumber, 500, 25, 0.9) +
             gaussian_peak(wavenumber, 800, 18, 1.1))
    
    # Component 3: Mineral C – peaks at 350, 700, 900 cm⁻¹
    comp3 = (lorentzian_peak(wavenumber, 350, 22, 0.8) +
             gaussian_peak(wavenumber, 700, 20, 1.0) +
             lorentzian_peak(wavenumber, 900, 16, 0.6))
    
    # Normalize each component to unit max
    comp1 = comp1 / comp1.max()
    comp2 = comp2 / comp2.max()
    comp3 = comp3 / comp3.max()
    
    pure_components = np.vstack([comp1, comp2, comp3])  # (3, n_points)
    n_components = 3
    
    # --- Generate mixed spectra ---
    # Random mixing proportions that sum to 1
    raw_weights = np.random.dirichlet(alpha=[2, 2, 2], size=n_mixtures)  # (n_mixtures, 3)
    true_weights = raw_weights.copy()
    
    # Mixed spectra (clean, no baseline, no noise)
    clean_spectra = true_weights @ pure_components  # (n_mixtures, n_points)
    
    # --- Add polynomial baseline drift ---
    x_norm = (wavenumber - wavenumber.mean()) / (wavenumber.max() - wavenumber.min())
    
    baselines = np.zeros((n_mixtures, n_points))
    for i in range(n_mixtures):
        # Random 3rd-order polynomial coefficients
        c0 = np.random.uniform(0.05, 0.15)
        c1 = np.random.uniform(-0.1, 0.1)
        c2 = np.random.uniform(0.05, 0.2)
        c3 = np.random.uniform(-0.05, 0.05)
        baselines[i] = c0 + c1 * x_norm + c2 * x_norm**2 + c3 * x_norm**3
    
    # --- Add Gaussian noise (SNR ~ snr) ---
    signal_power = np.mean(clean_spectra ** 2, axis=1, keepdims=True)
    noise_std = np.sqrt(signal_power / snr)
    noise = noise_std * np.random.randn(n_mixtures, n_points)
    
    # Final observed spectra
    observed_spectra = clean_spectra + baselines + noise
    
    print(f"Synthesized {n_mixtures} mixed spectra, {n_points} wavenumber points")
    print(f"True weights shape: {true_weights.shape}")
    print(f"Observed spectra shape: {observed_spectra.shape}")
    
    data = {
        'wavenumber': wavenumber,
        'pure_components': pure_components,
        'true_weights': true_weights,
        'clean_spectra': clean_spectra,
        'baselines': baselines,
        'observed_spectra': observed_spectra,
        'n_components': n_components,
    }
    
    return data
