import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def photoelectric_cross_section(E_keV):
    """Approximate photoelectric absorption cross-section in cm^2.
    Simplified Morrison & McCammon approximation."""
    return 2.0e-22 * (E_keV) ** (-8.0/3.0)

def absorbed_powerlaw(E_keV, gamma, K, N_H_1e22):
    """Absorbed power-law X-ray spectrum.
    Args:
        E_keV: energy bins in keV
        gamma: photon index (typically 1.5-3.0)
        K: normalization (photons/cm2/s/keV at 1 keV)
        N_H_1e22: hydrogen column density in units of 10^22 cm^-2
    Returns:
        photon flux in photons/cm2/s/keV
    """
    sigma = photoelectric_cross_section(E_keV)
    absorption = np.exp(-N_H_1e22 * 1e22 * sigma)
    return K * E_keV**(-gamma) * absorption

def forward_operator(params, E, dE, eff_area, exposure, background):
    """Predict observed counts for given spectral parameters.
    
    This is the forward model: parameters -> predicted observations.
    
    Args:
        params: tuple of (gamma, K, NH) - spectral parameters
        E: energy bin centers in keV
        dE: energy bin widths
        eff_area: effective area array
        exposure: exposure time in seconds
        background: background counts array
        
    Returns:
        predicted_counts: predicted photon counts per energy bin
    """
    gamma, K, NH = params
    
    # Compute absorbed power-law flux
    flux = absorbed_powerlaw(E, gamma, K, NH)
    
    # Convert flux to counts: counts = flux * area * dE * time + background
    predicted_counts = flux * eff_area * dE * exposure + background
    
    return predicted_counts
