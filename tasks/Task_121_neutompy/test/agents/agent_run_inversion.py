import numpy as np

import matplotlib

matplotlib.use("Agg")

from skimage.transform import radon, iradon

def run_inversion(sinogram, angles):
    """
    Run Filtered Backprojection (FBP) reconstruction.
    
    This implements the inverse of the Radon transform using FBP with Ram-Lak filter.
    
    FBP Algorithm:
    1. Apply ramp (Ram-Lak) filter to each projection in frequency domain
    2. Backproject filtered projections onto image grid
    3. Enforce non-negativity constraint (attenuation >= 0)
    
    Parameters
    ----------
    sinogram : ndarray
        Input sinogram (noisy line integral measurements)
    angles : ndarray
        Projection angles in degrees
        
    Returns
    -------
    recon : ndarray
        Reconstructed attenuation coefficient distribution
    """
    recon = iradon(sinogram, theta=angles, filter_name="ramp", circle=False)
    recon = np.maximum(recon, 0)  # physical: attenuation >= 0
    return recon
