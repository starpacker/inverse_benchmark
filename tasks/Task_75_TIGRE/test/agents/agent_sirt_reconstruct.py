import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import rotate as ndi_rotate

from skimage.transform import iradon

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(image, angles_deg):
    """
    Compute the Radon transform (sinogram) of a 2D image.
    Uses rotation + integration approach.
    
    Parameters:
    -----------
    image : ndarray
        2D input image of shape (n, n)
    angles_deg : ndarray
        1D array of projection angles in degrees
    
    Returns:
    --------
    sinogram : ndarray
        Sinogram of shape (n_angles, n_det)
    """
    n = image.shape[0]
    n_det = int(np.ceil(n * np.sqrt(2)))
    if n_det % 2 == 0:
        n_det += 1
    sinogram = np.zeros((len(angles_deg), n_det))

    # Pad image to n_det × n_det
    pad_total = n_det - n
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    img_padded = np.pad(image, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')

    for i, angle in enumerate(angles_deg):
        rotated = ndi_rotate(img_padded, -angle, reshape=False, order=1)
        sinogram[i, :] = rotated.sum(axis=0)[:n_det]

    return sinogram

def unfiltered_backproject(sinogram, angles_deg, img_size):
    """
    Unfiltered backprojection using iradon with filter_name=None.
    Used as the adjoint / correction operator in iterative methods.
    """
    sino_T = sinogram.T
    recon = iradon(sino_T, theta=angles_deg, filter_name=None,
                   output_size=img_size, circle=False)
    return recon

def sirt_reconstruct(sinogram, angles_deg, img_size, n_iter=50):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT).
    """
    recon = np.zeros((img_size, img_size))
    n_det = sinogram.shape[1]

    print(f"  SIRT: {n_iter} iterations ...")

    for it in range(n_iter):
        # Forward project current estimate
        sino_est = forward_operator(recon, angles_deg)

        # Trim to match
        sino_est_trim = sino_est[:, :n_det]

        # Residual
        residual = sinogram - sino_est_trim

        # Backproject residual (unfiltered — correct SIRT operator)
        correction = unfiltered_backproject(residual, angles_deg, img_size)

        # Update with relaxation
        recon += 0.1 * correction
        recon = np.maximum(recon, 0)

        if (it + 1) % 10 == 0:
            err = np.linalg.norm(residual)
            print(f"    iter {it+1:3d}: residual norm = {err:.4f}")

    return recon
