import numpy as np

import matplotlib

matplotlib.use('Agg')

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
