import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_116_thermoelastic"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def run_inversion(delta_T, thermo_coeff):
    """
    Inverse operator: Temperature → Stress Sum
    
    Direct algebraic inversion of the thermoelastic equation:
        Δ(σ₁ + σ₂) = −(ρ Cp)/(α T₀) × ΔT
    
    Parameters
    ----------
    delta_T : ndarray
        Measured temperature change (K)
    thermo_coeff : float
        Thermoelastic coefficient (α T₀)/(ρ Cp)
    
    Returns
    -------
    stress_sum : ndarray
        Recovered sum of principal stresses (Pa)
    """
    stress_sum = -delta_T / thermo_coeff
    return stress_sum
