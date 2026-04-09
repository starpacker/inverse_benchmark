import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

def run_inversion(sinograms, angles_deg, num_iterations=15, relaxation=0.02):
    """
    SART-like iterative reconstruction with FBP initialization.
    Refines FBP result by iteratively minimizing the sinogram residual.
    
    Args:
        sinograms: 3D numpy array of shape (D, num_det, num_angles)
        angles_deg: 1D array of projection angles in degrees
        num_iterations: number of SART iterations
        relaxation: relaxation parameter for SART update
        
    Returns:
        recon: 3D reconstructed volume of shape (D, N, N)
    """
    D = sinograms.shape[0]
    test = iradon(sinograms[0], theta=angles_deg, circle=True)
    N = test.shape[0]
    recon = np.zeros((D, N, N), dtype=np.float64)

    # Normalization volume: back-project a uniform sinogram
    ones_sino = np.ones_like(sinograms[0])
    norm_vol = iradon(ones_sino, theta=angles_deg, circle=True, filter_name=None)
    norm_vol = np.clip(np.abs(norm_vol), 1e-6, None)

    for iz in range(D):
        sino_meas = sinograms[iz]
        # FBP init
        r = iradon(sino_meas, theta=angles_deg, circle=True)
        r = np.clip(r, 0, None)

        for it in range(num_iterations):
            sino_est = radon(r, theta=angles_deg, circle=True)
            residual = sino_meas - sino_est
            bp_res = iradon(residual, theta=angles_deg, circle=True,
                            filter_name=None)
            r = r + relaxation * bp_res / norm_vol
            r = np.clip(r, 0, None)

        recon[iz] = r

    return recon
