import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(
    v_model: np.ndarray,
    gains: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the forward model: corrupt visibilities with gains and add noise.
    
    Forward model: V_obs_ij = g_i * V_model_ij * conj(g_j) + noise
    
    Parameters:
        v_model: Model visibilities (n_bl, n_freq, n_time)
        gains: Complex gains (n_ant, n_freq, n_time)
        ant1: First antenna indices for each baseline
        ant2: Second antenna indices for each baseline
        snr_db: Signal-to-noise ratio in dB
        rng: Random number generator
    
    Returns:
        v_obs: Observed (corrupted) visibilities (n_bl, n_freq, n_time)
    """
    n_bl, n_freq, n_time = v_model.shape
    
    v_obs = np.zeros_like(v_model)
    for bl_idx in range(n_bl):
        i, j = ant1[bl_idx], ant2[bl_idx]
        v_obs[bl_idx] = gains[i] * v_model[bl_idx] * np.conj(gains[j])
    
    # Add complex Gaussian noise
    signal_power = np.mean(np.abs(v_obs) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)
    noise = rng.normal(0, noise_std, v_obs.shape) + 1j * rng.normal(0, noise_std, v_obs.shape)
    v_obs += noise
    
    return v_obs
