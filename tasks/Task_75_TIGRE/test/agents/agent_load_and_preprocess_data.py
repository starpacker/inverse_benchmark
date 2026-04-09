import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import rotate as ndi_rotate

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_shepp_logan(size):
    """
    Generate the Shepp-Logan phantom.
    Standard CT test image with ellipses of varying attenuation.
    """
    img = np.zeros((size, size))
    Y, X = np.mgrid[:size, :size]
    X = (X - size / 2) / (size / 2)
    Y = (Y - size / 2) / (size / 2)

    ellipses = [
        (1.0,   0.69,  0.92,  0,      0,       0),
        (-0.8,  0.6624, 0.8740, 0,     -0.0184, 0),
        (-0.2,  0.1100, 0.3100, 0.22,  0,       -18),
        (-0.2,  0.1600, 0.4100, -0.22, 0,       18),
        (0.1,   0.2100, 0.2500, 0,     0.35,    0),
        (0.1,   0.0460, 0.0460, 0,     0.1,     0),
        (0.1,   0.0460, 0.0460, 0,     -0.1,    0),
        (0.1,   0.0460, 0.0230, -0.08, -0.605,  0),
        (0.1,   0.0230, 0.0230, 0,     -0.606,  0),
        (0.1,   0.0230, 0.0460, 0.06,  -0.605,  0),
    ]

    for A, a, b, x0, y0, phi_deg in ellipses:
        phi = np.radians(phi_deg)
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        x_rot = (X - x0) * cos_p + (Y - y0) * sin_p
        y_rot = -(X - x0) * sin_p + (Y - y0) * cos_p

        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        img[mask] += A

    img = np.clip(img, 0, None)
    return img

def load_and_preprocess_data(img_size, n_angles_sparse, noise_snr_db, seed):
    """
    Generate Shepp-Logan phantom and create sparse-view noisy sinogram.
    
    Parameters:
    -----------
    img_size : int
        Size of the phantom image (img_size x img_size)
    n_angles_sparse : int
        Number of projection angles for sparse-view CT
    noise_snr_db : float
        Signal-to-noise ratio in dB for additive noise
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    phantom : ndarray
        Ground truth Shepp-Logan phantom image
    sinogram_noisy : ndarray
        Noisy sparse-view sinogram
    angles_sparse : ndarray
        Array of projection angles in degrees
    """
    rng = np.random.default_rng(seed)
    
    # Generate phantom
    phantom = generate_shepp_logan(img_size)
    
    # Generate sparse angles
    angles_sparse = np.linspace(0, 180, n_angles_sparse, endpoint=False)
    
    # Compute clean sinogram using forward operator
    sinogram_sparse = forward_operator(phantom, angles_sparse)
    
    # Add Poisson-like noise
    sig_power = np.mean(sinogram_sparse**2)
    noise_power = sig_power / (10**(noise_snr_db / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(sinogram_sparse.shape)
    sinogram_noisy = sinogram_sparse + noise
    
    return phantom, sinogram_noisy, angles_sparse

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
