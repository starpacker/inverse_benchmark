import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.optimize import minimize

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_115_adapt_constitutive"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(strain, E, sigma_y, K, n):
    """
    Power-law hardening constitutive model (forward operator).
    
    Physics:
        σ = E × ε                    for ε ≤ ε_y  (elastic regime)
        σ = σ_y + K × (ε − ε_y)^n   for ε > ε_y  (plastic regime, power-law hardening)
        where ε_y = σ_y / E
    
    Args:
        strain: 1D array of strain values
        E: Young's modulus (MPa)
        sigma_y: Yield stress (MPa)
        K: Hardening coefficient (MPa)
        n: Hardening exponent
    
    Returns:
        stress: 1D array of predicted stress values
    """
    eps_y = sigma_y / E
    stress = np.empty_like(strain)
    elastic = strain <= eps_y
    plastic = ~elastic
    stress[elastic] = E * strain[elastic]
    eps_p = strain[plastic] - eps_y
    eps_p = np.maximum(eps_p, 0.0)
    stress[plastic] = sigma_y + K * np.power(eps_p, n)
    return stress

def run_inversion(strain, stress_obs):
    """
    Calibrate constitutive model parameters from noisy stress-strain data
    using L-BFGS-B optimization.
    
    The objective function minimizes:
        ||σ_obs − σ_model(params)||²
    
    Args:
        strain: 1D array of strain values
        stress_obs: 1D array of observed (noisy) stress values
    
    Returns:
        params_fit: fitted parameters [E, sigma_y, K, n]
        stress_recon: reconstructed stress using fitted parameters
    """
    
    def objective(params, strain_data, stress_data):
        """Sum-of-squares misfit."""
        E, sigma_y, K, n_exp = params
        stress_pred = forward_operator(strain_data, E, sigma_y, K, n_exp)
        return np.sum((stress_data - stress_pred) ** 2)
    
    # Estimate E from initial slope (first ~10% of data where elastic)
    n_init = max(5, len(strain) // 20)
    E_est = float(np.polyfit(strain[:n_init], stress_obs[:n_init], 1)[0])
    E_est = np.clip(E_est, 50e3, 500e3)
    
    # Estimate yield stress from the knee
    sy_est = float(stress_obs[n_init * 2]) if n_init * 2 < len(stress_obs) else 300.0
    
    x0 = [E_est, sy_est, 700.0, 0.4]
    bounds = [
        (50e3, 500e3),    # E
        (100.0, 800.0),   # σ_y
        (100.0, 2000.0),  # K
        (0.05, 0.95)      # n
    ]
    
    best_result = None
    best_cost = np.inf
    
    # Multi-start to avoid local minima
    starts = [
        x0,
        [E_est * 0.9, sy_est * 1.1, 800.0, 0.5],
        [E_est * 1.1, sy_est * 0.9, 600.0, 0.35],
        [200e3, 350.0, 800.0, 0.45],
    ]
    
    for s in starts:
        result = minimize(
            objective, s, args=(strain, stress_obs),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-18, "gtol": 1e-14}
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
    
    params_fit = best_result.x
    
    # Reconstruct stress with fitted parameters
    stress_recon = forward_operator(strain, *params_fit)
    
    return params_fit, stress_recon
