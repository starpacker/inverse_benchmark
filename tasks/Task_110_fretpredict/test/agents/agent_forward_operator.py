import numpy as np

import matplotlib

matplotlib.use("Agg")

def fret_efficiency(r, R0):
    """E(r) = 1 / (1 + (r/R0)^6)."""
    return 1.0 / (1.0 + (r / R0) ** 6)

def forward_operator(distances, R0, shot_noise_std=0.05):
    """
    Forward model: compute FRET efficiency from distances with shot noise.
    
    E(r) = 1 / (1 + (r/R0)^6) + noise
    
    Args:
        distances: array of inter-dye distances
        R0: Förster radius
        shot_noise_std: standard deviation of shot noise
        
    Returns:
        E_obs: observed FRET efficiencies clipped to [0, 1]
    """
    E_clean = fret_efficiency(distances, R0)
    noise = np.random.randn(len(E_clean)) * shot_noise_std
    E_obs = np.clip(E_clean + noise, 0.0, 1.0)
    return E_obs
