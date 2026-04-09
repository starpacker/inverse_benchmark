import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import nnls

from lmfit import Parameters, minimize as lm_minimize

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

def run_inversion(noisy_spectrum, metadata):
    """
    XRF spectrum deconvolution: recover elemental concentrations.
    
    Algorithm:
    1. Build element basis spectra (unit concentration)
    2. Non-negative least squares fitting: S = Σ c_k · B_k + background
    3. Refine with lmfit for better fit and uncertainties
    
    Parameters:
    -----------
    noisy_spectrum : ndarray
        Measured noisy spectrum
    metadata : dict
        Metadata containing energy grid, elements, detector params, etc.
    
    Returns:
    --------
    result : dict
        Reconstruction result containing:
        - nnls_concentrations: initial NNLS concentrations
        - refined_concentrations: refined concentrations from lmfit
        - recon_spectrum: reconstructed spectrum
        - fitted_background: fitted background spectrum
        - basis_spectra: element basis spectra
        - fit_result: lmfit result object
    """
    energy = metadata['energy']
    elements = metadata['elements']
    det_fwhm = metadata['det_fwhm']
    xrf_lines = metadata['xrf_lines']
    
    # Step 1: Build element basis spectra
    basis_spectra = build_xrf_model(energy, elements, det_fwhm, xrf_lines)
    
    # Step 2: Construct design matrix [B_1, B_2, ..., B_k, bg_exp, bg_const]
    n_elements = len(elements)
    n_energy = len(energy)
    
    # Design matrix: each column is an element's basis spectrum
    A = np.zeros((n_energy, n_elements + 2))
    for i, element in enumerate(elements):
        A[:, i] = basis_spectra[element]
    
    # Background basis functions
    A[:, n_elements] = np.exp(-2.5 * energy)  # exponential background
    A[:, n_elements + 1] = np.ones(n_energy)  # constant background
    
    # Step 3: Non-negative least squares
    coeffs, residual = nnls(A, noisy_spectrum)
    
    fitted_concentrations = {}
    for i, element in enumerate(elements):
        fitted_concentrations[element] = float(coeffs[i])
    
    bg_amp = coeffs[n_elements]
    bg_const = coeffs[n_elements + 1]
    
    print(f"[RECON] NNLS fitted concentrations: {fitted_concentrations}")
    
    # Step 4: Refine with lmfit for better fit and uncertainties
    params = Parameters()
    for i, element in enumerate(elements):
        params.add(f'c_{element}', value=coeffs[i], min=0)
    params.add('bg_amp', value=bg_amp * 500, min=0)
    params.add('bg_decay', value=2.5, min=0.5, max=5.0)
    params.add('bg_const', value=bg_const, min=0)
    
    def residual_func(params, energy, data, elements, basis_spectra):
        model = np.zeros_like(energy)
        for element in elements:
            model += params[f'c_{element}'].value * basis_spectra[element]
        model += params['bg_amp'].value * np.exp(-params['bg_decay'].value * energy)
        model += params['bg_const'].value
        return (data - model)
    
    result = lm_minimize(residual_func, params, args=(energy, noisy_spectrum, elements, basis_spectra))
    
    # Extract refined concentrations
    refined_concentrations = {}
    for element in elements:
        refined_concentrations[element] = float(result.params[f'c_{element}'].value)
    
    print(f"[RECON] Refined concentrations: {refined_concentrations}")
    
    # Build reconstructed spectrum
    recon_spectrum = np.zeros_like(energy)
    for element in elements:
        recon_spectrum += refined_concentrations[element] * basis_spectra[element]
    recon_spectrum += result.params['bg_amp'].value * np.exp(-result.params['bg_decay'].value * energy)
    recon_spectrum += result.params['bg_const'].value
    
    # Build fitted background
    fitted_bg = result.params['bg_amp'].value * np.exp(-result.params['bg_decay'].value * energy) + result.params['bg_const'].value
    
    return {
        'nnls_concentrations': fitted_concentrations,
        'refined_concentrations': refined_concentrations,
        'recon_spectrum': recon_spectrum,
        'fitted_background': fitted_bg,
        'basis_spectra': basis_spectra,
        'fit_result': result,
    }
