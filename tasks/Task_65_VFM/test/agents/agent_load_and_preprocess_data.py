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

def load_and_preprocess_data(gt_E, gt_nu, nx, ny, Lx, Ly, noise_strain, seed):
    """
    Generate synthetic full-field strain data for VFM identification.
    
    Parameters:
        gt_E: Ground truth Young's modulus (MPa)
        gt_nu: Ground truth Poisson's ratio
        nx, ny: Grid dimensions
        Lx, Ly: Specimen dimensions (mm)
        noise_strain: Noise level for strain measurements
        seed: Random seed
        
    Returns:
        Dictionary containing noisy strain fields, ground truth strain/stress,
        coordinates, and geometry parameters.
    """
    print("[DATA] Generating full-field strain data ...")
    
    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(dx/2, Lx - dx/2, nx)
    y = np.linspace(dy/2, Ly - dy/2, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Applied stress
    sigma_applied = 100.0  # MPa
    
    # Hole in plate (Kirsch solution approximation)
    cx, cy = Lx/2, Ly/2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    theta = np.arctan2(yy - cy, xx - cx)
    R = Ly / 6  # hole radius
    factor = sigma_applied / gt_E
    
    eps_xx = factor * (1 + (R/np.maximum(r, R*0.5))**2 *
                      (1.5*np.cos(2*theta) + np.cos(4*theta)) / 2)
    eps_yy = -gt_nu * eps_xx
    eps_xy = factor * (R/np.maximum(r, R*0.5))**2 * np.sin(2*theta) / 2
    
    # Compute stress from strain
    Q = plane_stress_stiffness(gt_E, gt_nu)
    sigma_xx = Q[0,0]*eps_xx + Q[0,1]*eps_yy
    sigma_yy = Q[1,0]*eps_xx + Q[1,1]*eps_yy
    sigma_xy = Q[2,2]*eps_xy
    
    # Add noise to strain measurements
    rng = np.random.default_rng(seed)
    eps_xx_n = eps_xx + noise_strain * rng.standard_normal((nx, ny))
    eps_yy_n = eps_yy + noise_strain * rng.standard_normal((nx, ny))
    eps_xy_n = eps_xy + noise_strain * rng.standard_normal((nx, ny))
    
    print(f"[DATA] ε_xx range: [{eps_xx.min():.6f}, {eps_xx.max():.6f}]")
    
    data = {
        'eps_xx_noisy': eps_xx_n,
        'eps_yy_noisy': eps_yy_n,
        'eps_xy_noisy': eps_xy_n,
        'eps_xx_gt': eps_xx,
        'eps_yy_gt': eps_yy,
        'eps_xy_gt': eps_xy,
        'sigma_xx_gt': sigma_xx,
        'sigma_yy_gt': sigma_yy,
        'sigma_xy_gt': sigma_xy,
        'xx': xx,
        'yy': yy,
        'Lx': Lx,
        'Ly': Ly,
        'nx': nx,
        'ny': ny,
        'gt_E': gt_E,
        'gt_nu': gt_nu
    }
    
    return data
