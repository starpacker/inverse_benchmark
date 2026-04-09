import numpy as np

import matplotlib

matplotlib.use('Agg')

def gaussian_peak(energy, center, amplitude, sigma):
    """Gaussian peak profile for detector-broadened XRF line."""
    return amplitude * np.exp(-0.5 * ((energy - center) / sigma) ** 2)

def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def generate_element_spectrum(energy, element, concentration, det_sigma, xrf_lines):
    """
    Generate XRF spectrum for a single element.
    Each characteristic line is a Gaussian broadened by detector resolution.
    """
    spectrum = np.zeros_like(energy)
    if element not in xrf_lines:
        return spectrum
    
    for line_name, line_energy, rel_intensity in xrf_lines[element]:
        amplitude = concentration * rel_intensity
        spectrum += gaussian_peak(energy, line_energy, amplitude, det_sigma)
    
    return spectrum

def forward_operator(concentrations, energy, det_fwhm, xrf_lines):
    """
    Forward model: concentrations → XRF spectrum (without background)
    
    S(E) = Σ_k c_k · Σ_lines G(E; E_line, σ_det) · I_rel
    
    Parameters:
    -----------
    concentrations : dict
        Element concentrations {element: concentration}
    energy : ndarray
        Energy grid (keV)
    det_fwhm : float
        Detector FWHM (keV)
    xrf_lines : dict
        XRF line data {element: [(line_name, energy, rel_intensity), ...]}
    
    Returns:
    --------
    spectrum : ndarray
        Predicted XRF spectrum
    """
    det_sigma = fwhm_to_sigma(det_fwhm)
    spectrum = np.zeros_like(energy)
    
    for element, conc in concentrations.items():
        spectrum += generate_element_spectrum(energy, element, conc, det_sigma, xrf_lines)
    
    return spectrum
