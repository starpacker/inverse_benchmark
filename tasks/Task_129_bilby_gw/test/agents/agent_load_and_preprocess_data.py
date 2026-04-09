import os

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")

import bilby

def load_and_preprocess_data(
    injection_parameters,
    duration,
    sampling_freq,
    minimum_freq,
    reference_freq,
    approximant,
    outdir,
):
    """
    Load and preprocess gravitational wave data.
    
    Sets up interferometers, injects a CBC signal with known parameters,
    and prepares the waveform generator.
    
    Returns:
        dict containing:
            - ifos: InterferometerList with injected signal
            - waveform_generator: bilby WaveformGenerator
            - gt_strain_fd: ground truth frequency-domain strain (H1)
            - noisy_strain_fd: noisy frequency-domain strain (H1)
            - freq_array: frequency array
            - true_chirp_mass: computed chirp mass
            - true_mass_ratio: computed mass ratio
            - h1: H1 interferometer reference
    """
    # Setup bilby logger
    os.makedirs(outdir, exist_ok=True)
    bilby.core.utils.setup_logger(outdir=outdir, label="bilby_gw", log_level="WARNING")
    
    # Compute derived true parameters
    m1 = injection_parameters["mass_1"]
    m2 = injection_parameters["mass_2"]
    true_chirp_mass = (m1 * m2) ** (3.0 / 5) / (m1 + m2) ** (1.0 / 5)
    true_mass_ratio = m2 / m1
    
    print(f"True chirp_mass = {true_chirp_mass:.4f}, mass_ratio = {true_mass_ratio:.4f}")
    
    # Waveform arguments
    waveform_arguments = dict(
        waveform_approximant=approximant,
        reference_frequency=reference_freq,
        minimum_frequency=minimum_freq,
    )
    
    # Create waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_freq,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    
    # Setup interferometers
    print("[1/7] Setting up interferometers with injected signal...")
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_freq,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - duration + 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
    )
    
    # Compute ground-truth signal
    print("[2/7] Computing ground-truth waveform...")
    polarisations = waveform_generator.frequency_domain_strain(injection_parameters)
    freq_array = waveform_generator.frequency_array
    h1 = ifos[0]
    gt_strain_fd = h1.get_detector_response(polarisations, injection_parameters)
    noisy_strain_fd = h1.strain_data.frequency_domain_strain.copy()
    
    return {
        "ifos": ifos,
        "waveform_generator": waveform_generator,
        "gt_strain_fd": gt_strain_fd,
        "noisy_strain_fd": noisy_strain_fd,
        "freq_array": freq_array,
        "true_chirp_mass": true_chirp_mass,
        "true_mass_ratio": true_mass_ratio,
        "h1": h1,
    }
