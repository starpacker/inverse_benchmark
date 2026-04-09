import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import rotate as ndi_rotate

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
