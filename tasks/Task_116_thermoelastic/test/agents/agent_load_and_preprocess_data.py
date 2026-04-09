import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_116_thermoelastic"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def load_and_preprocess_data(a_hole, plate_r, nr, ntheta, sigma_0, noise_level, seed,
                              alpha, t0, rho, cp):
    """
    Load and preprocess data for thermoelastic stress analysis.
    
    Creates the polar grid, computes the ground truth stress field using
    the Kirsch solution, generates clean temperature change via thermoelastic
    effect, and adds noise to simulate measurements.
    
    Parameters
    ----------
    a_hole : float
        Hole radius (m)
    plate_r : float
        Outer radius for grid (m)
    nr : int
        Number of radial grid points
    ntheta : int
        Number of angular grid points
    sigma_0 : float
        Far-field stress (Pa)
    noise_level : float
        Noise level as fraction of max |ΔT|
    seed : int
        Random seed for reproducibility
    alpha : float
        Coefficient of thermal expansion (1/K)
    t0 : float
        Reference temperature (K)
    rho : float
        Density (kg/m³)
    cp : float
        Specific heat (J/(kg·K))
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'R': radial coordinate meshgrid
        - 'THETA': angular coordinate meshgrid
        - 'gt_stress_sum': ground truth stress sum field (Pa)
        - 'delta_T_noisy': noisy temperature change (K)
        - 'thermo_coeff': thermoelastic coefficient
    """
    np.random.seed(seed)
    
    # Compute thermoelastic coefficient
    thermo_coeff = alpha * t0 / (rho * cp)
    
    # Create polar grid (exclude hole interior)
    r = np.linspace(a_hole * 1.01, plate_r, nr)
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    R, THETA = np.meshgrid(r, theta, indexing="ij")
    
    # Compute Kirsch stress field (ground truth)
    # Kirsch solution for infinite plate with circular hole under uniaxial tension
    ratio = a_hole / R
    ratio2 = ratio ** 2
    ratio4 = ratio ** 4
    cos2t = np.cos(2 * THETA)
    
    sigma_rr = (sigma_0 / 2) * ((1 - ratio2) + (1 - 4 * ratio2 + 3 * ratio4) * cos2t)
    sigma_tt = (sigma_0 / 2) * ((1 + ratio2) - (1 + 3 * ratio4) * cos2t)
    
    # Stress sum (sum of principal stresses in 2D)
    gt_stress_sum = sigma_rr + sigma_tt
    
    # Compute clean temperature change using thermoelastic equation
    # ΔT = −(α T₀)/(ρ Cp) × Δ(σ₁ + σ₂)
    delta_T_clean = -thermo_coeff * gt_stress_sum
    
    # Add noise to simulate real measurements
    noise = noise_level * np.max(np.abs(delta_T_clean)) * np.random.randn(*delta_T_clean.shape)
    delta_T_noisy = delta_T_clean + noise
    
    data_dict = {
        'R': R,
        'THETA': THETA,
        'gt_stress_sum': gt_stress_sum,
        'delta_T_noisy': delta_T_noisy,
        'thermo_coeff': thermo_coeff
    }
    
    return data_dict
