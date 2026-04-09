import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(c_model, G, c0):
    """
    Compute travel-time perturbations from a sound-speed model.
    
    Forward model: δt = G @ (s - s0), where s = 1/c (slowness).
    
    Args:
        c_model: 2D sound speed array (nx, ny)
        G: ray-path kernel matrix (sparse)
        c0: background sound speed
        
    Returns:
        dt: travel-time perturbations (n_rays,)
        t_abs: absolute travel times (n_rays,)
    """
    s = 1.0 / c_model.ravel()
    s0 = 1.0 / c0
    ds = s - s0
    dt = G @ ds
    t_abs = G @ s
    return dt, t_abs
