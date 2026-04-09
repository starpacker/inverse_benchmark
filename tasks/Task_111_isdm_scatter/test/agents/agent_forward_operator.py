import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(x):
    """
    Forward model: Compute Fourier magnitude from object.
    
    In scattering imaging, the measured data is |F[object]|^2 (power spectrum)
    obtained from speckle autocorrelation. The forward operator maps the object
    to its Fourier magnitude.
    
    Args:
        x: Object image (2D numpy array)
    
    Returns:
        y_pred: Fourier magnitude |F[x]|
    """
    F_x = np.fft.fft2(x)
    y_pred = np.abs(F_x)
    return y_pred
