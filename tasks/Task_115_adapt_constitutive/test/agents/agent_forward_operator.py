import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

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
