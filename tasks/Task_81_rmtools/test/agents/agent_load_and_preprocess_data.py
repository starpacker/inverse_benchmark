import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(
    freq_min_mhz,
    freq_max_mhz,
    n_channels,
    noise_sigma_jy,
    components,
    stokes_i_flux,
    random_seed=42
):
    """
    Generate synthetic broadband polarization observations.
    
    Args:
        freq_min_mhz: Minimum frequency in MHz
        freq_max_mhz: Maximum frequency in MHz
        n_channels: Number of frequency channels
        noise_sigma_jy: Noise level in Jy/beam per channel
        components: List of dicts with 'phi', 'amplitude', 'chi0' for each source
        stokes_i_flux: Total intensity (Jy)
        random_seed: Random seed for reproducibility
    
    Returns:
        observations: dict with freq_hz, lambda_sq, Q, U, dQ, dU, I, dI
        ground_truth: dict with Q_clean, U_clean, components
    """
    np.random.seed(random_seed)
    
    # Frequency array
    freq_hz = np.linspace(freq_min_mhz * 1e6, freq_max_mhz * 1e6, n_channels)
    c = 2.998e8
    lambda_sq = (c / freq_hz) ** 2
    
    # Forward model: compute clean polarization
    phi_vals = [comp['phi'] for comp in components]
    amp_vals = [comp['amplitude'] for comp in components]
    chi0_vals = [comp['chi0'] for comp in components]
    
    P_clean, _ = forward_operator(phi_vals, amp_vals, chi0_vals, freq_hz)
    
    # Extract Q and U
    Q_clean = P_clean.real
    U_clean = P_clean.imag
    
    # Add noise
    Q_noisy = Q_clean + np.random.normal(0, noise_sigma_jy, n_channels)
    U_noisy = U_clean + np.random.normal(0, noise_sigma_jy, n_channels)
    
    # Stokes I (flat spectrum)
    I_arr = np.ones(n_channels) * stokes_i_flux
    dI_arr = np.ones(n_channels) * noise_sigma_jy * 0.1  # I noise much lower
    
    # Noise arrays
    dQ_arr = np.ones(n_channels) * noise_sigma_jy
    dU_arr = np.ones(n_channels) * noise_sigma_jy
    
    observations = {
        'freq_hz': freq_hz,
        'lambda_sq': lambda_sq,
        'Q': Q_noisy,
        'U': U_noisy,
        'dQ': dQ_arr,
        'dU': dU_arr,
        'I': I_arr,
        'dI': dI_arr,
    }
    
    ground_truth = {
        'Q_clean': Q_clean,
        'U_clean': U_clean,
        'components': components,
    }
    
    print(f"  [FORWARD] Frequency range: {freq_min_mhz}-{freq_max_mhz} MHz, "
          f"{n_channels} channels")
    print(f"  [FORWARD] λ² range: [{lambda_sq.min():.6f}, {lambda_sq.max():.6f}] m²")
    print(f"  [FORWARD] Components: {len(components)}")
    for i, c in enumerate(components):
        print(f"    Component {i+1}: φ={c['phi']:.1f} rad/m², "
              f"A={c['amplitude']:.2f} Jy, χ₀={c['chi0']:.2f} rad")
    
    return observations, ground_truth

def forward_operator(phi_values, amplitudes, chi0_values, freq_hz):
    """
    Forward model: Faraday depth spectrum → complex polarization P(λ²).
    
    P(λ²) = Σ_i A_i · exp(2i·(χ0_i + φ_i·λ²))
    
    For Faraday-thin components:
        Q(ν) + iU(ν) = Σ_i A_i · exp(2i·φ_i·λ² + 2i·χ0_i)
    
    Args:
        phi_values: Faraday depths (rad/m²) - list or array
        amplitudes: Component amplitudes (Jy) - list or array
        chi0_values: Initial polarization angles (rad) - list or array
        freq_hz: Observation frequencies (Hz) - numpy array
    
    Returns:
        P_complex: Complex polarization spectrum (Q + iU) - numpy array
        lambda_sq: Wavelength squared array (m²) - numpy array
    """
    c = 2.998e8  # speed of light m/s
    lambda_sq = (c / freq_hz) ** 2  # λ² in m²
    
    P = np.zeros(len(freq_hz), dtype=complex)
    for phi, amp, chi0 in zip(phi_values, amplitudes, chi0_values):
        P += amp * np.exp(2j * (chi0 + phi * lambda_sq))
    
    return P, lambda_sq
