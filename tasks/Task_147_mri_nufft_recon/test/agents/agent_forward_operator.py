import matplotlib

matplotlib.use('Agg')

import sys

import os

import warnings

import numpy as np

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')

sys.path.insert(0, REPO_DIR)

warnings.filterwarnings('ignore', message='Samples will be rescaled')

def forward_operator(x, op_plain):
    """
    Apply the NUFFT forward operator to an image.
    
    Computes y = F_nu * x, where F_nu is the Non-Uniform FFT
    at radial trajectory positions.
    
    Parameters
    ----------
    x : ndarray (N, N)
        Input image (real or complex)
    op_plain : NUFFT operator
        The NUFFT operator without density compensation
        
    Returns
    -------
    y_pred : ndarray
        Predicted k-space data at non-Cartesian positions
    """
    x_complex = x.astype(np.complex64)
    y_pred = op_plain.op(x_complex)
    return y_pred
