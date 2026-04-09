import numpy as np

import scipy.fft

def backproject(sinogram, theta):
    """
    Explicit Backprojection algorithm.
    """
    num_angles, num_detectors = sinogram.shape
    N = num_detectors
    recon = np.zeros((N, N), dtype=np.float32)
    
    for i, angle in enumerate(theta):
        projection = sinogram[i]
        tiled_projection = np.tile(projection, (N, 1))
        
        # Rotate it back to the original angle
        rotated = scipy.ndimage.rotate(tiled_projection, -angle, reshape=False, order=1, mode='constant', cval=0.0)
        recon += rotated
        
    # Scale factor approximation
    return recon * (np.pi / (2 * num_angles))
