import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")

def forward_operator(
    params,
    waveform_generator,
    detector,
):
    """
    Forward operator: compute detector response for given source parameters.
    
    Maps source parameters theta -> predicted strain h(f).
    
    Args:
        params: dict of CBC source parameters
        waveform_generator: bilby WaveformGenerator
        detector: bilby Interferometer (e.g., H1)
    
    Returns:
        predicted_strain_fd: complex frequency-domain strain array
    """
    # Generate frequency-domain polarisations from source parameters
    polarisations = waveform_generator.frequency_domain_strain(params)
    
    # Apply detector antenna pattern and time/phase shifts
    predicted_strain_fd = detector.get_detector_response(polarisations, params)
    
    return predicted_strain_fd
