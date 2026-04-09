import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from skimage.transform import radon, iradon, resize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def forward_operator(x, theta):
    """
    Radon transform (forward projection) using skimage.
    
    Args:
        x: Input image (N x N)
        theta: Projection angles
        
    Returns:
        y_pred: Sinogram (forward projection of x)
    """
    return radon(x, theta=theta, circle=True)
