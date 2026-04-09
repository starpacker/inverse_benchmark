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

def build_xrf_model(energy, elements, det_fwhm, xrf_lines):
    """
    Build the multi-element XRF fitting model.
    Pre-compute element basis spectra (unit concentration).
    """
    det_sigma = fwhm_to_sigma(det_fwhm)
    
    basis_spectra = {}
    for element in elements:
        basis_spectra[element] = generate_element_spectrum(energy, element, 1.0, det_sigma, xrf_lines)
    
    return basis_spectra
