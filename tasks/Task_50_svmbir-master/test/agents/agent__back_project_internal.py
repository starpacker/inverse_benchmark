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

def _back_project_internal(sinogram, angles, image_shape):
    """
    Adjoint of the forward projector.
    """
    if HAS_SKIMAGE:
        # skimage iradon takes (num_det, num_angles)
        sino_T = sinogram.T
        theta_deg = np.degrees(angles)
        # filter_name=None implies unfiltered backprojection (adjoint of Radon)
        recon = iradon(sino_T, theta=theta_deg, output_size=image_shape[0], filter_name=None, circle=True)
        return recon
    else:
        # Manual backprojection
        num_rows, num_cols = image_shape
        backproj = np.zeros(image_shape, dtype=np.float32)
        for i, angle_rad in enumerate(angles):
            angle_deg = np.degrees(angle_rad)
            projection = sinogram[i, :]
            smeared = np.tile(projection, (num_rows, 1))
            rotated_back = scipy.ndimage.rotate(smeared, -angle_deg, reshape=False, order=1, mode='constant', cval=0.0)
            backproj += rotated_back
        return backproj
