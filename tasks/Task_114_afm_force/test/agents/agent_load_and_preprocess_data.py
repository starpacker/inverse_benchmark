import numpy as np

import matplotlib

matplotlib.use("Agg")

def load_and_preprocess_data(z_min, z_max, n_points, epsilon, sigma, noise_level, seed):
    """
    Generate the distance grid, compute ground truth Lennard-Jones force,
    and add noise to simulated frequency shift data.
    
    Parameters:
    -----------
    z_min : float
        Minimum tip-sample distance (m)
    z_max : float
        Maximum distance (m)
    n_points : int
        Number of distance points
    epsilon : float
        Depth of LJ potential well (J)
    sigma : float
        Zero-crossing distance (m)
    noise_level : float
        Fractional noise level on frequency shift
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    z : ndarray
        Distance grid (m)
    F_gt : ndarray
        Ground truth Lennard-Jones force (N)
    noise_level : float
        Noise level for later use
    seed : int
        Seed for later use
    epsilon : float
        LJ parameter
    sigma : float
        LJ parameter
    """
    np.random.seed(seed)
    
    # Distance grid
    z = np.linspace(z_min, z_max, n_points)
    
    # Ground truth Lennard-Jones force:
    # F(z) = -dU/dz = 24ε/σ × [2(σ/z)^13 - (σ/z)^7]
    ratio = sigma / z
    F_gt = 24 * epsilon / sigma * (2 * ratio**13 - ratio**7)
    
    return z, F_gt, noise_level, seed
