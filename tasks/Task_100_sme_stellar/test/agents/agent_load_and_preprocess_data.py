import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.special import voigt_profile

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_100_sme_stellar"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def load_and_preprocess_data(wave_min, wave_max, n_wave, gt_teff, gt_logg, gt_feh, 
                              gt_abundances, line_list, snr, seed=42):
    """
    Load and preprocess data: generate wavelength grid, ground-truth spectrum, 
    and noisy observed spectrum.
    
    Parameters
    ----------
    wave_min : float
        Minimum wavelength in Angstroms
    wave_max : float
        Maximum wavelength in Angstroms
    n_wave : int
        Number of wavelength points
    gt_teff : float
        Ground-truth effective temperature (K)
    gt_logg : float
        Ground-truth surface gravity log(cm/s^2)
    gt_feh : float
        Ground-truth metallicity [Fe/H]
    gt_abundances : dict
        Ground-truth element abundances {element: [X/H]}
    line_list : list
        List of (element, rest_wavelength, gf_value) tuples
    snr : float
        Signal-to-noise ratio
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - wavelength: array of wavelength values
        - flux_gt: ground-truth flux spectrum
        - flux_obs: noisy observed spectrum
        - gt_params: tuple of (T_eff, log_g, feh, abundances)
        - line_list: the line list used
    """
    np.random.seed(seed)
    
    # Generate wavelength grid
    wavelength = np.linspace(wave_min, wave_max, n_wave)
    
    # Generate ground-truth spectrum using forward operator
    flux_gt = forward_operator(
        wavelength=wavelength,
        T_eff=gt_teff,
        log_g=gt_logg,
        feh=gt_feh,
        abundances=gt_abundances,
        line_list=line_list
    )
    
    # Add noise
    noise_level = flux_gt.mean() / snr
    noise = np.random.normal(0, noise_level, n_wave)
    flux_obs = flux_gt + noise
    
    gt_params = (gt_teff, gt_logg, gt_feh, gt_abundances)
    
    data_dict = {
        "wavelength": wavelength,
        "flux_gt": flux_gt,
        "flux_obs": flux_obs,
        "gt_params": gt_params,
        "line_list": line_list,
        "n_wave": n_wave
    }
    
    return data_dict

def forward_operator(wavelength, T_eff, log_g, feh, abundances, line_list):
    """
    Forward model: synthesize a stellar spectrum.
    
    Physics:
    - Planck continuum based on effective temperature
    - Voigt absorption lines for each element
    - Line depth depends on abundance and oscillator strength
    - Gaussian width depends on thermal broadening (T_eff)
    - Lorentzian width depends on pressure broadening (log_g)
    
    Parameters
    ----------
    wavelength : array
        Wavelength grid in Angstroms
    T_eff : float
        Effective temperature in K
    log_g : float
        Surface gravity log(cm/s^2)
    feh : float
        Metallicity [Fe/H]
    abundances : dict
        Element abundances {element: [X/H]}
    line_list : list
        List of (element, rest_wavelength, gf_value) tuples
        
    Returns
    -------
    flux : array
        Normalized flux spectrum
    """
    # Compute Planck continuum
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    lam_m = wavelength * 1e-10
    B = (2.0 * h * c**2 / lam_m**5) / (np.exp(h * c / (lam_m * k * T_eff)) - 1.0)
    continuum = B / B.max()
    
    # Compute absorption from all lines
    absorption = np.zeros_like(wavelength)
    
    for elem, lam0, gf in line_list:
        ab = abundances.get(elem, 0.0) + feh
        
        # Gaussian width depends on T_eff (thermal broadening)
        # Using approximate atomic mass of 56 AMU (Fe-like)
        sigma_thermal = lam0 / 3e5 * np.sqrt(2.0 * 1.381e-23 * T_eff / (56.0 * 1.66e-27))
        sigma_G = max(sigma_thermal, 0.02)
        
        # Lorentzian width depends on log_g (pressure broadening)
        gamma_L = 0.05 * 10.0**(0.3 * (4.5 - log_g))
        
        # Line depth depends on abundance and oscillator strength
        depth = 0.3 * 10.0**(gf + ab) / (1.0 + 10.0**(gf + ab))
        depth = np.clip(depth, 0.0, 0.95)
        
        # Voigt profile for this line
        delta = wavelength - lam0
        profile = voigt_profile(delta, sigma_G, gamma_L)
        if profile.max() > 0:
            profile = profile / profile.max()
        line_absorption = depth * profile
        
        absorption += line_absorption
    
    # Final flux = continuum × (1 - absorption)
    flux = continuum * (1.0 - np.clip(absorption, 0, 0.99))
    
    return flux
