import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(x, mask):
    """
    Forward operator for MRI: applies 2D FFT and undersampling mask.
    
    Models the acquisition: y = M * F * x
    where M is the undersampling mask and F is the 2D FFT.
    
    Args:
        x: Image in spatial domain (NxN array)
        mask: Undersampling mask (NxN array)
    
    Returns:
        y_pred: Predicted k-space measurements (undersampled)
    """
    x_kspace = np.fft.fft2(x, norm='ortho')
    y_pred = mask * x_kspace
    return y_pred
