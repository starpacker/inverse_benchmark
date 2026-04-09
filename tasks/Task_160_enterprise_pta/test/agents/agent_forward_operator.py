import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def powerlaw_psd(freqs, log10_A, gamma):
    """Power-law power spectral density: S(f) = A^2/(12*pi^2) * (f/f_yr)^(-gamma) * f_yr^(-3)."""
    A = 10.0 ** log10_A
    f_yr = 1.0 / (365.25 * 86400.0)
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs / f_yr) ** (-gamma) * f_yr ** (-3)

def forward_operator(params, data_dict):
    """
    Compute the predicted power spectral densities given model parameters.
    
    This is the forward model that maps parameters to observables (PSDs).
    
    Args:
        params: Array [log10_A_gw, log10_A_red, gamma_red]
        data_dict: Dictionary containing freqs and other data
        
    Returns:
        psd_dict: Dictionary containing predicted red noise and GW PSDs
    """
    log10_A_gw, log10_A_red, gamma_red = params
    freqs = data_dict['freqs']
    
    # Compute predicted PSDs
    psd_red_pred = powerlaw_psd(freqs, log10_A_red, gamma_red)
    psd_gw_pred = powerlaw_psd(freqs, log10_A_gw, 13.0 / 3.0)
    
    return {
        'psd_red': psd_red_pred,
        'psd_gw': psd_gw_pred,
        'freqs': freqs
    }
