import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_104_bisip_sip"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(freq, rho0, m, tau, c):
    """
    Cole-Cole complex resistivity forward model.
    ρ*(ω) = ρ_0 × [1 - m × (1 - 1/(1 + (iωτ)^c))]
    
    Parameters:
        freq: ndarray, frequency array (Hz)
        rho0: float, DC resistivity (Ohm·m)
        m: float, chargeability (0-1)
        tau: float, time constant (s)
        c: float, frequency exponent (0-1)
    
    Returns:
        rho_star: ndarray, complex resistivity
    """
    omega = 2.0 * np.pi * freq
    z = (1j * omega * tau) ** c
    rho_star = rho0 * (1.0 - m * (1.0 - 1.0 / (1.0 + z)))
    return rho_star
