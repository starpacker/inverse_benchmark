import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(slip_vec, G):
    """
    Forward operator: compute surface displacements from slip distribution.
    
    Implements: d = G * s
    where:
    - d: displacement vector (3*N_obs,) containing (ux, uy, uz) for each station
    - G: Green's function matrix (3*N_obs, N_patches)
    - s: slip vector (N_patches,)
    
    The Green's functions are computed using Okada (1985) analytical solutions
    for rectangular dislocation in an elastic half-space.
    
    Args:
        slip_vec: 1D array of slip values for each fault patch (N_patches,)
        G: Green's function matrix (3*N_obs, N_patches)
    
    Returns:
        d_pred: predicted displacement vector (3*N_obs,)
    """
    d_pred = G @ slip_vec
    return d_pred
