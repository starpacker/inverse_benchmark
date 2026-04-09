import numpy as np

import scipy.ndimage

try:
    from skimage.transform import radon, iradon
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using slower fallback implementations.")

HAS_SKIMAGE = True

def forward_operator(x, angles):
    """
    Computes the Radon Transform (Forward Projection).
    
    Args:
        x (np.array): Input image (N, N).
        angles (np.array): Projection angles in radians.
        
    Returns:
        y_pred (np.array): Sinogram (num_angles, num_detectors).
    """
    if HAS_SKIMAGE:
        theta_deg = np.degrees(angles)
        # skimage returns (num_det, num_angles), we want (num_angles, num_det)
        sino = radon(x, theta=theta_deg, circle=True)
        return sino.T
    else:
        num_angles = len(angles)
        num_rows, num_cols = x.shape
        num_det = num_cols 
        sinogram = np.zeros((num_angles, num_det), dtype=np.float32)
        
        for i, angle_rad in enumerate(angles):
            angle_deg = np.degrees(angle_rad)
            rotated_img = scipy.ndimage.rotate(x, angle_deg, reshape=False, order=1, mode='constant', cval=0.0)
            sinogram[i, :] = rotated_img.sum(axis=0)
        return sinogram
