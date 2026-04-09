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

def sirt_reconstruct(sinogram, theta, n_iter=10):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT).
    """
    num_angles, num_detectors = sinogram.shape
    N = num_detectors
    
    recon = np.zeros((N, N), dtype=np.float32)
    
    # Calculate Row Sums (R)
    ones_img = np.ones((N, N), dtype=np.float32)
    row_sums = radon_transform_logic(ones_img, theta)
    row_sums[row_sums == 0] = 1.0
    
    # Calculate Column Sums (C)
    ones_sino = np.ones_like(sinogram)
    col_sums = backproject(ones_sino, theta)
    col_sums[col_sums == 0] = 1.0
    
    for k in range(n_iter):
        fp = radon_transform_logic(recon, theta)
        diff = sinogram - fp
        correction_term = diff / row_sums
        correction = backproject(correction_term, theta)
        
        recon += correction / col_sums
        recon[recon < 0] = 0
        
    return recon
