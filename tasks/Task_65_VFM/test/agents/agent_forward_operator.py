import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def plane_stress_stiffness(E, nu):
    """Plane stress stiffness matrix Q."""
    factor = E / (1 - nu**2)
    Q = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    return Q

def forward_operator(params, eps_xx, eps_yy, eps_xy):
    """
    Compute stress field from strain field using plane stress constitutive law.
    
    Parameters:
        params: Tuple or array (E, nu) - material parameters
        eps_xx, eps_yy, eps_xy: Strain field components
        
    Returns:
        sigma_xx, sigma_yy, sigma_xy: Stress field components
    """
    E, nu = params
    
    # Compute plane stress stiffness matrix
    Q = plane_stress_stiffness(E, nu)
    
    # Compute stress from strain using σ = Q·ε
    sigma_xx = Q[0,0]*eps_xx + Q[0,1]*eps_yy
    sigma_yy = Q[1,0]*eps_xx + Q[1,1]*eps_yy
    sigma_xy = Q[2,2]*eps_xy
    
    return sigma_xx, sigma_yy, sigma_xy
