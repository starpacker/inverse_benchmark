import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.optimize import minimize

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

def run_inversion(data_dict):
    """Fit spectrum using maximum likelihood optimization for Poisson data.
    
    Args:
        data_dict: dictionary from load_and_preprocess_data containing:
            - E_centers, dE, observed, eff_area, exposure, background
            
    Returns:
        result_dict: dictionary containing:
            - gamma_fit: fitted photon index
            - K_fit: fitted normalization
            - NH_fit: fitted column density
            - recovered_counts: predicted counts with fitted parameters
    """
    E = data_dict['E_centers']
    dE = data_dict['dE']
    observed = data_dict['observed']
    eff_area = data_dict['eff_area']
    exposure = data_dict['exposure']
    background = data_dict['background']
    
    def neg_log_likelihood(opt_params):
        """Negative Poisson log-likelihood for optimization."""
        gamma, logK, logNH = opt_params
        K = 10**logK
        NH = 10**logNH
        
        # Use forward operator to get predicted counts
        predicted = forward_operator((gamma, K, NH), E, dE, eff_area, exposure, background)
        predicted = np.maximum(predicted, 1e-10)
        
        # Poisson log-likelihood: sum(obs*log(pred) - pred)
        return -np.sum(observed * np.log(predicted) - predicted)

    # Initial guesses (in log space for K and NH)
    x0 = [2.0, np.log10(0.005), np.log10(0.3)]
    bounds = [(1.0, 4.0), (-4, 0), (-2, 2)]

    # Run L-BFGS-B optimization
    opt_result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')

    # Extract fitted parameters
    gamma_fit = opt_result.x[0]
    K_fit = 10**opt_result.x[1]
    NH_fit = 10**opt_result.x[2]

    # Compute recovered counts using forward operator
    recovered_counts = forward_operator((gamma_fit, K_fit, NH_fit), E, dE, eff_area, exposure, background)

    result_dict = {
        'gamma_fit': gamma_fit,
        'K_fit': K_fit,
        'NH_fit': NH_fit,
        'recovered_counts': recovered_counts,
        'optimization_success': opt_result.success
    }
    
    return result_dict
