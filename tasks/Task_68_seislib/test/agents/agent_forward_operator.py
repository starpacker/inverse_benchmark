import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(G, dm_flat, c0):
    """
    Compute travel-time residuals using straight-ray approximation.
    
    Forward Model:
        δt = G @ (δm / c0)
    where G is the ray-path kernel matrix and δm is the slowness perturbation.
    
    Args:
        G: sparse kernel matrix (n_rays, n_cells)
        dm_flat: flattened velocity perturbation model (n_cells,)
        c0: reference phase velocity [km/s]
    
    Returns:
        dt: travel-time residuals (n_rays,)
    """
    dt = G @ (dm_flat / c0)
    return dt
