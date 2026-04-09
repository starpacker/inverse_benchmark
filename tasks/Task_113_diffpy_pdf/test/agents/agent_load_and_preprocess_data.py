import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_113_diffpy_pdf"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def fcc_neighbor_distances(a, r_max, max_shell=200):
    """
    Compute interatomic distances and coordination numbers for FCC structure.

    For FCC, neighbor distances are a × sqrt(n/2) for certain n values.
    Returns list of (distance, coordination_number) pairs.
    """
    distances = []
    n_max = int(np.ceil(r_max / a)) + 1
    for h in range(-n_max, n_max + 1):
        for k in range(-n_max, n_max + 1):
            for l in range(-n_max, n_max + 1):
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (h + bx) * a
                    y = (k + by) * a
                    z = (l + bz) * a
                    d = np.sqrt(x**2 + y**2 + z**2)
                    if 0.1 < d < r_max:
                        distances.append(d)

    distances = np.sort(distances)
    shells = []
    tol = 0.01
    i = 0
    while i < len(distances) and len(shells) < max_shell:
        d_ref = distances[i]
        count = 0
        while i < len(distances) and abs(distances[i] - d_ref) < tol:
            count += 1
            i += 1
        shells.append((d_ref, count))

    return shells

def load_and_preprocess_data(r_min, r_max, dr, a_true, B_true, scale_true, noise_level, seed):
    """
    Generate r grid and synthetic PDF data with noise.
    
    Parameters:
    -----------
    r_min : float
        Minimum r value in Angstroms
    r_max : float
        Maximum r value in Angstroms
    dr : float
        Step size in Angstroms
    a_true : float
        True lattice constant
    B_true : float
        True Debye-Waller factor
    scale_true : float
        True scale factor
    noise_level : float
        Fractional noise level
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    r : ndarray
        r grid values
    G_gt : ndarray
        Ground truth G(r)
    G_noisy : ndarray
        Noisy measured G(r)
    params_true : dict
        Dictionary of true parameters
    """
    np.random.seed(seed)
    
    r = np.arange(r_min, r_max, dr)
    
    shells = fcc_neighbor_distances(a_true, r_max)
    sigma = np.sqrt(B_true)
    
    G_gt = np.zeros_like(r)
    rho0 = 4 / a_true**3
    
    for d_n, coord_n in shells:
        sigma_n = sigma * np.sqrt(1 + 0.002 * d_n**2)
        amplitude = coord_n / (4 * np.pi * d_n**2 * rho0)
        peak = amplitude * np.exp(-0.5 * ((r - d_n) / sigma_n)**2) / (sigma_n * np.sqrt(2 * np.pi))
        G_gt += peak
    
    G_gt = scale_true * G_gt / (np.max(np.abs(G_gt)) + 1e-12)
    G_gt *= np.exp(-0.01 * r**2)
    
    noise = noise_level * np.max(np.abs(G_gt)) * np.random.randn(len(r))
    G_noisy = G_gt + noise
    
    params_true = {
        'a': a_true,
        'B': B_true,
        'scale': scale_true,
        'r_max': r_max
    }
    
    return r, G_gt, G_noisy, params_true
