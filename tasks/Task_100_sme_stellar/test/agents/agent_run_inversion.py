import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.special import voigt_profile

from scipy.optimize import differential_evolution

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_100_sme_stellar"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

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

def run_inversion(data_dict, bounds_teff=(4500, 7000), bounds_logg=(3.0, 5.5),
                  bounds_feh=(-1.0, 0.5), bounds_abundance=(-0.5, 0.5),
                  maxiter=300, seed=123):
    """
    Run inverse problem: fit stellar parameters using differential evolution.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data containing:
        - wavelength: wavelength array
        - flux_obs: observed flux
        - line_list: line list
        - gt_params: ground-truth parameters (for element list)
    bounds_teff : tuple
        Bounds for effective temperature
    bounds_logg : tuple
        Bounds for surface gravity
    bounds_feh : tuple
        Bounds for metallicity
    bounds_abundance : tuple
        Bounds for individual element abundances
    maxiter : int
        Maximum iterations for optimizer
    seed : int
        Random seed for optimizer
        
    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - T_eff: fitted effective temperature
        - log_g: fitted surface gravity
        - feh: fitted metallicity
        - abundances: fitted abundances dict
        - flux_fit: fitted spectrum
        - fit_params: tuple of fitted parameters
    """
    wavelength = data_dict["wavelength"]
    flux_obs = data_dict["flux_obs"]
    line_list = data_dict["line_list"]
    gt_params = data_dict["gt_params"]
    gt_abundances = gt_params[3]
    
    # Get sorted element list
    elements = sorted(gt_abundances.keys())
    
    # Build bounds
    bounds = [bounds_teff, bounds_logg, bounds_feh]
    for _ in elements:
        bounds.append(bounds_abundance)
    
    def objective(vec):
        """Chi-squared objective function."""
        T_eff = vec[0]
        log_g = vec[1]
        feh = vec[2]
        abundances = {elem: vec[3 + i] for i, elem in enumerate(elements)}
        
        flux_model = forward_operator(
            wavelength=wavelength,
            T_eff=T_eff,
            log_g=log_g,
            feh=feh,
            abundances=abundances,
            line_list=line_list
        )
        
        residual = flux_obs - flux_model
        return np.sum(residual**2)
    
    # Run differential evolution
    result = differential_evolution(
        objective, bounds,
        seed=seed, maxiter=maxiter, tol=1e-8, popsize=20,
        mutation=(0.5, 1.5), recombination=0.9,
        polish=True
    )
    
    # Unpack results
    T_fit = result.x[0]
    logg_fit = result.x[1]
    feh_fit = result.x[2]
    ab_fit = {elem: result.x[3 + i] for i, elem in enumerate(elements)}
    
    # Compute fitted spectrum
    flux_fit = forward_operator(
        wavelength=wavelength,
        T_eff=T_fit,
        log_g=logg_fit,
        feh=feh_fit,
        abundances=ab_fit,
        line_list=line_list
    )
    
    fit_params = (T_fit, logg_fit, feh_fit, ab_fit)
    
    result_dict = {
        "T_eff": T_fit,
        "log_g": logg_fit,
        "feh": feh_fit,
        "abundances": ab_fit,
        "flux_fit": flux_fit,
        "fit_params": fit_params
    }
    
    return result_dict
