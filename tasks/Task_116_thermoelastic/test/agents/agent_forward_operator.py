import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_116_thermoelastic"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(stress_sum, thermo_coeff):
    """
    Forward operator: Stress → Temperature Change
    
    Implements the thermoelastic equation:
        ΔT = −(α T₀)/(ρ Cp) × Δ(σ₁ + σ₂)
    
    Parameters
    ----------
    stress_sum : ndarray
        Sum of principal stresses σ₁ + σ₂ (Pa)
    thermo_coeff : float
        Thermoelastic coefficient (α T₀)/(ρ Cp)
    
    Returns
    -------
    delta_T : ndarray
        Temperature change (K)
    """
    delta_T = -thermo_coeff * stress_sum
    return delta_T
