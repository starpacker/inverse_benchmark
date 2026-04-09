import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

import pyxu.operator as pxo

def forward_operator(x, img_shape, kernel):
    """
    Apply the forward convolution operator H to input x.
    
    Parameters:
    -----------
    x : np.ndarray
        Input image (can be 1D flattened or 2D)
    img_shape : tuple
        Shape of the 2D image (height, width)
    kernel : np.ndarray
        2D Gaussian blur kernel
        
    Returns:
    --------
    y_pred : np.ndarray
        Blurred output (1D flattened)
    """
    # Build Pyxu Convolve operator
    H = pxo.Convolve(
        arg_shape=img_shape,
        kernel=kernel,
        center=(kernel.shape[0]//2, kernel.shape[1]//2),
        mode="constant",
    )
    
    # Ensure x is flattened
    x_flat = x.ravel()
    
    # Apply forward operator
    y_pred = H(x_flat)
    
    return y_pred
