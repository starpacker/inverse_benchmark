import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_115_adapt_constitutive"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def load_and_preprocess_data(
    strain_max: float = 0.15,
    n_points: int = 500,
    noise_level: float = 0.02,
    seed: int = 42,
    E_true: float = 210e3,
    sy_true: float = 350.0,
    K_true: float = 800.0,
    n_true: float = 0.45
):
    """
    Generate synthetic stress-strain data for constitutive model calibration.
    
    Returns:
        strain: 1D array of strain values
        stress_gt: 1D array of ground-truth stress (clean)
        stress_noisy: 1D array of noisy stress observations
        params_true: list of true parameters [E, sigma_y, K, n]
    """
    np.random.seed(seed)
    
    # Create strain grid
    strain = np.linspace(0, strain_max, n_points)
    
    # Compute ground-truth stress using power-law hardening model
    eps_y = sy_true / E_true
    stress_gt = np.empty_like(strain)
    elastic = strain <= eps_y
    plastic = ~elastic
    stress_gt[elastic] = E_true * strain[elastic]
    eps_p = strain[plastic] - eps_y
    eps_p = np.maximum(eps_p, 0.0)
    stress_gt[plastic] = sy_true + K_true * np.power(eps_p, n_true)
    
    # Add Gaussian noise
    noise = noise_level * np.max(np.abs(stress_gt)) * np.random.randn(len(strain))
    stress_noisy = stress_gt + noise
    
    params_true = [E_true, sy_true, K_true, n_true]
    
    return strain, stress_gt, stress_noisy, params_true
