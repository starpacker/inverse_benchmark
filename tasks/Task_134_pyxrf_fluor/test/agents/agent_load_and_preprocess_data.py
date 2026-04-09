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

def generate_background(energy, a=500.0, b=2.5, c=0.1):
    """
    Generate Bremsstrahlung background: exponential + scatter.
    B(E) = a * exp(-b*E) + c
    """
    return a * np.exp(-b * energy) + c

def load_and_preprocess_data(e_min, e_max, e_step, detector_fwhm, sample_elements, 
                              gt_concentrations, xrf_lines, scale_factor=10.0, seed=42):
    """
    Generate synthetic XRF spectrum with known elemental concentrations.
    
    Parameters:
    -----------
    e_min : float
        Minimum energy (keV)
    e_max : float
        Maximum energy (keV)
    e_step : float
        Energy step (keV)
    detector_fwhm : float
        Detector FWHM (keV)
    sample_elements : list
        List of element symbols to include
    gt_concentrations : dict
        Ground truth concentrations {element: concentration}
    xrf_lines : dict
        XRF line data {element: [(line_name, energy, rel_intensity), ...]}
    scale_factor : float
        Scale factor for count rates
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    noisy_spectrum : ndarray
        Noisy measured spectrum
    gt_array : ndarray
        Ground truth concentrations as array
    metadata : dict
        Additional data including energy grid, clean spectrum, etc.
    """
    np.random.seed(seed)
    
    energy = np.arange(e_min, e_max, e_step)
    det_sigma = fwhm_to_sigma(detector_fwhm)
    
    # Generate clean element spectra using forward model
    clean_signal = np.zeros_like(energy)
    for element, conc in gt_concentrations.items():
        clean_signal += generate_element_spectrum(energy, element, conc, det_sigma, xrf_lines)
    
    # Generate background
    background = generate_background(energy)
    
    # Total clean spectrum
    clean_spectrum = clean_signal + background
    
    # Add Poisson-like noise (photon counting statistics)
    expected_counts = clean_spectrum * scale_factor
    noisy_counts = np.random.poisson(np.maximum(expected_counts, 1).astype(int)).astype(float)
    noisy_spectrum = noisy_counts / scale_factor
    
    # GT as array
    gt_array = np.array([gt_concentrations[el] for el in sample_elements])
    
    metadata = {
        'energy': energy,
        'clean_signal': clean_signal,
        'background': background,
        'clean_spectrum': clean_spectrum,
        'elements': sample_elements,
        'gt_concentrations': gt_concentrations,
        'det_fwhm': detector_fwhm,
        'xrf_lines': xrf_lines,
    }
    
    print(f"[DATA] Generated XRF spectrum with {len(sample_elements)} elements")
    print(f"[DATA] Energy range: {energy[0]:.1f} - {energy[-1]:.2f} keV")
    print(f"[DATA] GT concentrations: {gt_concentrations}")
    
    return noisy_spectrum, gt_array, metadata
