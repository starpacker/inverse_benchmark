import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def forward_operator(output):
    """
    Forward: Extract photospheric B_r from PFSS solution.
    
    Given the 3D magnetic field solution B(r,θ,φ), extract the 
    radial component at the photosphere (r = R_sun).
    
    Args:
        output: PFSS output object containing the 3D magnetic field solution
    
    Returns:
        br_photosphere: 2D numpy array of B_r at the photosphere
    """
    # Get the magnetic field at the inner boundary
    br_photosphere_raw = output.bc[0][:, :, 0]  # B_r at inner boundary
    br_photosphere = np.array(br_photosphere_raw.value if hasattr(br_photosphere_raw, 'value') else br_photosphere_raw).T
    return br_photosphere
