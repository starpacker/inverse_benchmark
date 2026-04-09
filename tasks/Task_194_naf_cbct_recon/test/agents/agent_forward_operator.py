import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

def forward_operator(volume, angles_deg):
    """
    Slice-wise 2D Radon transform (cone-beam projection approximation).
    
    Args:
        volume: 3D numpy array of shape (D, H, W)
        angles_deg: 1D array of projection angles in degrees
        
    Returns:
        sinograms: 3D numpy array of shape (D, num_det, num_angles)
    """
    D = volume.shape[0]
    test = radon(volume[0], theta=angles_deg, circle=True)
    sinograms = np.zeros((D, test.shape[0], len(angles_deg)), dtype=np.float64)
    for iz in range(D):
        sinograms[iz] = radon(volume[iz], theta=angles_deg, circle=True)
    return sinograms
