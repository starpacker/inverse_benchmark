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

def run_inversion(br_map, nr, rss):
    """
    PFSS reconstruction: solve Laplace's equation for the coronal field.
    
    Solves ∇²Ψ = 0 with boundary conditions:
    - Inner boundary (r = R_sun): B_r = -∂Ψ/∂r from magnetogram
    - Outer boundary (r = R_ss): B_r = 0 (purely radial at source surface)
    
    Solution via spherical harmonic decomposition.
    
    Args:
        br_map: sunpy Map of photospheric magnetogram
        nr: Number of radial grid points
        rss: Source surface radius (in solar radii)
    
    Returns:
        output: PFSS output object containing the 3D magnetic field solution
        br_recon: 2D numpy array of reconstructed B_r at photosphere
    """
    import pfsspy
    
    print(f"  [PFSS] Creating input: nr={nr}, rss={rss}")
    pfss_input = pfsspy.Input(br_map, nr, rss)
    
    print("  [PFSS] Solving PFSS equations...")
    output = pfsspy.pfss(pfss_input)
    
    print(f"  [PFSS] Solution grid: {output.bg.shape}")
    print(f"  [PFSS] Source surface radius: {rss} R_sun")
    
    # Extract reconstructed B_r at photosphere using forward operator
    br_recon = forward_operator(output)
    
    return output, br_recon
