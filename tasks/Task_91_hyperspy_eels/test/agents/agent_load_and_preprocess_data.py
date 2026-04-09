import matplotlib

matplotlib.use('Agg')

import numpy as np

from numpy.fft import fft, ifft

def load_and_preprocess_data(n_channels, de, zlp_fwhm, t_over_lambda, noise_level, random_seed=42):
    """
    Generate synthetic EELS data including ZLP, ground truth SSD, and measured spectrum.
    
    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    de : float
        Energy per channel (eV).
    zlp_fwhm : float
        Full width at half maximum of ZLP in eV.
    t_over_lambda : float
        Relative thickness parameter.
    noise_level : float
        Relative noise amplitude.
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    data : dict
        Dictionary containing:
        - 'zlp': Zero-loss peak array
        - 'ssd_norm': Normalized single scattering distribution
        - 'gt_ssd': Ground truth SSD (t/lambda * ssd_norm)
        - 'measured': Measured spectrum with noise
        - 'energy_axis': Energy axis in eV
        - 't_over_lambda': Relative thickness
        - 'n_channels': Number of channels
        - 'de': Energy dispersion
    """
    # Generate energy axis
    energy_axis = np.arange(n_channels) * de
    
    # Generate Zero-Loss Peak (ZLP) as a Gaussian centered at channel 0
    sigma_ch = (zlp_fwhm / de) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    idx = np.arange(n_channels)
    dist = np.minimum(idx, n_channels - idx).astype(float)
    zlp = np.exp(-dist**2 / (2.0 * sigma_ch**2))
    zlp = zlp / np.sum(zlp)  # Normalize to sum = 1
    
    # Generate ground truth Single Scattering Distribution (SSD)
    # Simulates typical EELS plasmon features for aluminum-like material
    peaks = [
        (7.0, 2.0, 0.30),   # surface plasmon: (center_eV, sigma_eV, amplitude)
        (15.0, 3.0, 0.80),  # bulk plasmon (dominant)
        (30.0, 5.0, 0.20),  # interband transitions
        (50.0, 4.0, 0.10),  # weak high-energy feature
    ]
    
    ssd = np.zeros(n_channels)
    for center_eV, sigma_eV, amp in peaks:
        c = center_eV / de
        s = sigma_eV / de
        ssd += amp * np.exp(-(idx - c)**2 / (2.0 * s**2))
    
    # Enforce causality: zero for E < 2 eV and E > 100 eV
    ssd[idx < int(2.0 / de)] = 0.0
    ssd[idx > int(100.0 / de)] = 0.0
    
    ssd_norm = ssd / (np.sum(ssd) + 1e-30)
    
    # Ground truth: what the deconvolution should recover
    gt_ssd = t_over_lambda * ssd_norm
    
    # Generate measured spectrum using forward model
    Z = fft(zlp)
    S = fft(ssd_norm)
    J_ft = Z * np.exp(t_over_lambda * S)
    measured_clean = np.real(ifft(J_ft))
    measured_clean = np.maximum(measured_clean, 0.0)
    
    # Add Gaussian noise
    rng = np.random.default_rng(random_seed)
    noise = noise_level * np.max(measured_clean) * rng.standard_normal(n_channels)
    measured = np.maximum(measured_clean + noise, 0.0)
    
    data = {
        'zlp': zlp,
        'ssd_norm': ssd_norm,
        'gt_ssd': gt_ssd,
        'measured': measured,
        'energy_axis': energy_axis,
        't_over_lambda': t_over_lambda,
        'n_channels': n_channels,
        'de': de
    }
    
    return data
