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

def load_and_preprocess_data(seed=42):
    """Generate and preprocess synthetic X-ray spectrum with Poisson noise.
    
    Args:
        seed: random seed for reproducibility
        
    Returns:
        data_dict: dictionary containing:
            - E_centers: energy bin centers in keV
            - dE: energy bin widths
            - observed: observed photon counts (Poisson sampled)
            - expected_counts: expected source counts (no background)
            - background: background counts
            - eff_area: effective area array
            - exposure: exposure time in seconds
            - true_params: dict with true gamma, K, N_H values
    """
    np.random.seed(seed)
    
    # Energy grid: 0.5 - 10 keV, 200 bins
    E_edges = np.linspace(0.5, 10.0, 201)
    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    dE = np.diff(E_edges)

    # True parameters
    true_gamma = 1.8
    true_K = 0.01  # photons/cm2/s/keV at 1 keV
    true_NH = 0.5  # 10^22 cm^-2

    # Effective area (simplified, ~1000 cm^2 at peak)
    eff_area = 1000.0 * np.exp(-0.1 * (E_centers - 2.0)**2)

    # Exposure time
    exposure = 50000.0  # 50 ks

    # Expected counts
    flux = absorbed_powerlaw(E_centers, true_gamma, true_K, true_NH)
    expected_counts = flux * eff_area * dE * exposure

    # Add background (flat, low level)
    background = 0.5 * dE * exposure * eff_area / 1000.0

    total_expected = expected_counts + background

    # Poisson sampling
    observed = np.random.poisson(np.maximum(total_expected, 0.1))

    data_dict = {
        'E_centers': E_centers,
        'dE': dE,
        'observed': observed,
        'expected_counts': expected_counts,
        'background': background,
        'eff_area': eff_area,
        'exposure': exposure,
        'true_params': {'gamma': true_gamma, 'K': true_K, 'N_H': true_NH}
    }
    
    return data_dict
