import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

def forward_operator(x, theta):
    """
    Apply the forward operator (Radon transform) to an image.
    
    The Radon transform computes line integrals of an image along specified
    angles, producing a sinogram.
    
    Args:
        x: Input 2D image (numpy array)
        theta: Array of projection angles in degrees
        
    Returns:
        y_pred: Sinogram (2D numpy array), shape (n_detectors, n_angles)
    """
    y_pred = radon(x, theta=theta, circle=True)
    return y_pred
