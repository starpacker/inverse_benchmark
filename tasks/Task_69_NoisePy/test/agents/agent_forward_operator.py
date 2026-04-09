import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(data):
    """
    Compute forward travel-time perturbations using straight-ray approximation.
    
    Forward Model:
        δt = G @ δs, where δs = -dm/c0 (slowness perturbation)
    
    Args:
        data: dict containing G (kernel matrix), dm_gt_flat, c0, noise_level, rng
        
    Returns:
        dict with clean and noisy travel time perturbations
    """
    G = data['G']
    dm_gt_flat = data['dm_gt_flat']
    c0 = data['c0']
    noise_level = data['noise_level']
    rng = data['rng']
    
    # dm is δc/c0, slowness perturbation δs ≈ -dm/c0
    ds = -dm_gt_flat / c0
    dt_clean = G @ ds
    
    # Add noise
    noise = noise_level * np.std(dt_clean) * rng.standard_normal(len(dt_clean))
    dt_noisy = dt_clean + noise
    
    return {
        'dt_clean': dt_clean,
        'dt_noisy': dt_noisy,
        'noise': noise
    }
