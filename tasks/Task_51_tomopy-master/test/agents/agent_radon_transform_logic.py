import numpy as np

import scipy.fft

def radon_transform_logic(image, theta):
    """
    Explicit implementation of the Radon Transform (Forward Projector).
    """
    num_angles = len(theta)
    N = image.shape[1] 
    sinogram = np.zeros((num_angles, N), dtype=np.float32)

    for i, angle in enumerate(theta):
        # Rotate the image. order=1 (linear)
        rotated = scipy.ndimage.rotate(image, -angle, reshape=False, order=1, mode='constant', cval=0.0)
        sinogram[i] = rotated.sum(axis=0)
        
    return sinogram
