import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def gaussian_peak(wavelengths, center, width, amplitude=1.0):
    """Generate a Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def create_pure_spectra(wavelengths):
    """Create 3 pure component spectra with Gaussian peaks."""
    n_wl = len(wavelengths)
    S = np.zeros((3, n_wl))

    # S1: peaks at 250 nm and 450 nm
    S[0] = gaussian_peak(wavelengths, 250, 20, 1.0) + \
           gaussian_peak(wavelengths, 450, 25, 0.8)

    # S2: peaks at 350 nm and 550 nm
    S[1] = gaussian_peak(wavelengths, 350, 22, 0.9) + \
           gaussian_peak(wavelengths, 550, 28, 1.0)

    # S3: peaks at 300 nm, 500 nm, and 650 nm
    S[2] = gaussian_peak(wavelengths, 300, 18, 0.7) + \
           gaussian_peak(wavelengths, 500, 20, 0.85) + \
           gaussian_peak(wavelengths, 650, 22, 0.6)

    return S

def create_concentration_profiles(n_samples, n_components):
    """Create sinusoidal concentration profiles (simulating kinetics)."""
    t = np.linspace(0, 2 * np.pi, n_samples)
    C = np.zeros((n_samples, n_components))

    # Component 1: decaying sinusoidal
    C[:, 0] = 0.5 * (1 + np.sin(t)) * np.exp(-0.2 * t) + 0.1

    # Component 2: growing then decaying
    C[:, 1] = 0.8 * np.sin(t + np.pi / 3) ** 2 + 0.05

    # Component 3: delayed growth
    C[:, 2] = 0.6 * (1 - np.exp(-0.5 * t)) * np.abs(np.cos(t / 2)) + 0.1

    # Ensure non-negativity
    C = np.maximum(C, 0)
    return C

def load_and_preprocess_data(n_samples, n_components, n_wavelengths, snr_db):
    """
    Generate synthetic spectral data for MCR-ALS decomposition.
    
    This function creates:
    - Pure component spectra (S_true)
    - Concentration profiles (C_true)
    - Clean data matrix (D_clean = C_true @ S_true)
    - Noisy data matrix (D_noisy = D_clean + noise)
    
    Parameters
    ----------
    n_samples : int
        Number of samples (rows in concentration matrix)
    n_components : int
        Number of chemical components
    n_wavelengths : int
        Number of wavelength points
    snr_db : float
        Signal-to-noise ratio in decibels
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'wavelengths': wavelength array
        - 'S_true': true pure component spectra (n_components, n_wavelengths)
        - 'C_true': true concentration profiles (n_samples, n_components)
        - 'D_clean': noise-free data matrix (n_samples, n_wavelengths)
        - 'D_noisy': noisy data matrix (n_samples, n_wavelengths)
        - 'snr_db': signal-to-noise ratio
        - 'actual_snr': actual computed SNR
        - 'n_components': number of components
    """
    wavelengths = np.linspace(200, 700, n_wavelengths)
    
    # Create ground truth spectra and concentrations
    S_true = create_pure_spectra(wavelengths)
    C_true = create_concentration_profiles(n_samples, n_components)
    
    # Generate clean data matrix D = C @ S
    D_clean = C_true @ S_true
    
    # Add Gaussian noise based on SNR
    signal_power = np.mean(D_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), D_clean.shape)
    D_noisy = D_clean + noise
    
    # Compute actual SNR
    actual_snr = 10 * np.log10(np.mean(D_clean**2) / np.mean((D_noisy - D_clean)**2))
    
    print("=" * 60)
    print("MCR-ALS Spectral Decomposition")
    print("=" * 60)
    print(f"Pure spectra matrix S: {S_true.shape}")
    print(f"Concentration matrix C: {C_true.shape}")
    print(f"Data matrix D: {D_noisy.shape}")
    print(f"SNR: {snr_db} dB")
    print(f"Actual SNR: {actual_snr:.2f} dB")
    
    return {
        'wavelengths': wavelengths,
        'S_true': S_true,
        'C_true': C_true,
        'D_clean': D_clean,
        'D_noisy': D_noisy,
        'snr_db': snr_db,
        'actual_snr': actual_snr,
        'n_components': n_components
    }
